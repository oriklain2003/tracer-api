"""
AI routes - chat, AI analyze, reasoning endpoints.
"""
from __future__ import annotations

import base64
import json
import logging
import sqlite3
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import psycopg2.extras

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI"])

# Anomaly rules metadata - same as in flights.py
RULES_METADATA = [
    # Emergency & Safety (Red)
    {"id": 1, "name": "Emergency Squawks", "nameHe": "קודי חירום", "description": "Transponder emergency code (7500, 7600, 7700)", "category": "emergency", "color": "red"},
    {"id": 2, "name": "Crash", "nameHe": "התרסקות", "description": "Aircraft crash or suspected crash event", "category": "emergency", "color": "red"},
    {"id": 3, "name": "Proximity Alert", "nameHe": "התראת קרבה", "description": "Dangerous proximity between aircraft", "category": "emergency", "color": "red"},
    
    # Flight Operations (Blue)
    {"id": 4, "name": "Holding Pattern", "nameHe": "דפוס המתנה", "description": "Aircraft in holding pattern", "category": "flight_ops", "color": "blue"},
    {"id": 5, "name": "Go Around", "nameHe": "גו-אראונד", "description": "Aborted landing and go-around maneuver", "category": "flight_ops", "color": "blue"},
    {"id": 6, "name": "Return to Land", "nameHe": "חזרה לנחיתה", "description": "Aircraft returning to departure airport", "category": "flight_ops", "color": "blue"},
    {"id": 7, "name": "Unplanned Landing", "nameHe": "נחיתה לא מתוכננת", "description": "Landing at unplanned airport", "category": "flight_ops", "color": "blue"},
    
    # Technical (Purple)
    {"id": 8, "name": "Signal Loss", "nameHe": "אובדן אות", "description": "Loss of ADS-B signal", "category": "technical", "color": "purple"},
    {"id": 9, "name": "Off Course", "nameHe": "סטייה ממסלול", "description": "Significant deviation from expected flight path", "category": "technical", "color": "purple"},
    {"id": 14, "name": "GPS Jamming", "nameHe": "שיבוש GPS", "description": "GPS jamming indicators detected", "category": "technical", "color": "purple"},
    
    # Military (Green)
    {"id": 10, "name": "Military Flight", "nameHe": "טיסה צבאית", "description": "Identified military aircraft", "category": "military", "color": "green"},
    {"id": 11, "name": "Operational Military", "nameHe": "טיסה צבאית מבצעית", "description": "Military aircraft on operational mission", "category": "military", "color": "green"},
    {"id": 12, "name": "Suspicious Behavior", "nameHe": "התנהגות חשודה", "description": "Unusual or suspicious flight behavior", "category": "military", "color": "green"},
    {"id": 13, "name": "Flight Academy", "nameHe": "בית ספר לטיסה", "description": "Training flight from flight school", "category": "military", "color": "green"},
]

# These will be set by the main api.py module
FEEDBACK_TAGGED_DB_PATH: Path = None
FEEDBACK_DB_PATH: Path = None

# Function references from api.py
_get_unified_track = None
_fetch_flight_details = None
_format_flight_summary_for_llm = None
_generate_flight_map_image = None
_rewrite_triggers_with_feedback = None
_openai_client = None
_gemini_client = None
_types = None  # google.genai.types


def configure(
        feedback_tagged_db_path: Path,
        feedback_db_path: Path,
        get_unified_track_func,
        fetch_flight_details_func,
        format_flight_summary_for_llm_func,
        generate_flight_map_image_func,
        rewrite_triggers_with_feedback_func,
        openai_client,
        gemini_client,
        genai_types,
):
    """Configure the router with paths and dependencies from api.py"""
    global FEEDBACK_TAGGED_DB_PATH, FEEDBACK_DB_PATH
    global _get_unified_track, _fetch_flight_details, _format_flight_summary_for_llm
    global _generate_flight_map_image, _rewrite_triggers_with_feedback
    global _openai_client, _gemini_client, _types

    FEEDBACK_TAGGED_DB_PATH = feedback_tagged_db_path
    FEEDBACK_DB_PATH = feedback_db_path
    _get_unified_track = get_unified_track_func
    _fetch_flight_details = fetch_flight_details_func
    _format_flight_summary_for_llm = format_flight_summary_for_llm_func
    _generate_flight_map_image = generate_flight_map_image_func
    _rewrite_triggers_with_feedback = rewrite_triggers_with_feedback_func
    _openai_client = openai_client
    _gemini_client = gemini_client
    _types = genai_types


# System prompts
CHAT_SYSTEM_PROMPT = """You are an advanced aviation analysis assistant specialized in flight anomaly detection.
You analyze flight paths and provide insights about anomalies, patterns, and potential issues.
Use the flight summary, track data, and anomaly analysis provided to give accurate and helpful responses.
When referencing specific points or segments, use the provided indices from KEY_POINTS section.
Be concise but thorough in your explanations."""

AI_COPILOT_SYSTEM_PROMPT = """
## IDENTITY AND PURPOSE

You are FiveAir Copilot, an elite aviation anomaly assistant embedded within a live tactical map interface. Your mission is to analyze flight paths in the Eastern Mediterranean (Israel, Lebanon, Syria, Jordan, Cyprus, Egypt) and provide clear, professional, and immediate situational awareness.

---

## VISUAL INTERPRETATION RULES

- **Green Marker:** Represents the start of the tracked segment or the entry point.
- **Red Marker:** Represents the current aircraft position, the stop point, or the exit point.
- **Airport Proximity:** If a marker is near a known airport, assume takeoff/landing.

---

## OUTPUT FORMAT STRUCTURE

**Summary:** 1-2 sentences capturing the real story.
**Situation Analysis:** Detailed breakdown of aircraft behavior.
**Main Issue:** State the core anomaly directly.
**Confidence:** One short line (e.g., High, Medium, Low).

---

## MAP HIGHLIGHTING ACTIONS

When you want to point to something on the map, you MAY output a JSON action block.
The JSON must be wrapped in triple backticks with the json language tag.

Available Actions:
1. Highlight a specific point: `{"action": "highlight_point", "lat": 32.1234, "lon": 34.9876}`
2. Highlight a segment: `{"action": "highlight_segment", "startIndex": 120, "endIndex": 150}`
3. Focus on a specific time: `{"action": "focus_time", "timestamp": 1702216324}`

Use AT MOST one or two actions per response.
"""


class ChatRequest(BaseModel):
    flight_time: int
    messages: List[Dict[str, str]]
    flight_id: str
    analysis: Optional[Dict[str, Any]] = None
    points: Optional[List[Dict[str, Any]]] = None
    user_question: str


class ProximityContext(BaseModel):
    """Context about proximity events with other aircraft."""
    other_flight_id: Optional[str] = None
    other_callsign: Optional[str] = None
    distance_nm: Optional[float] = None
    altitude_diff_ft: Optional[float] = None
    timestamp: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class AIAnalyzeRequest(BaseModel):
    screenshot: Optional[str] = None
    question: str
    flight_id: str
    flight_data: List[Dict[str, Any]]
    anomaly_report: Optional[Dict[str, Any]] = None
    selected_point: Optional[Dict[str, Any]] = None
    flight_time: Optional[int] = None
    history: List[Dict[str, str]] = []
    length: Optional[str] = 'medium'
    language: Optional[str] = 'en'
    # Proximity context - other aircraft involved in proximity events
    proximity_context: Optional[List[ProximityContext]] = None
    # Mode: 'fast' uses gemini-2.5-flash, 'thinking' uses gemini-3-pro-preview
    mode: Optional[str] = 'thinking'


class ReasoningRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    flight_id: Optional[str] = None
    points: Optional[List[Dict[str, Any]]] = None
    anomaly_report: Optional[Dict[str, Any]] = None


class ClassifyAnomalyRequest(BaseModel):
    """Request for classifying an anomaly flight."""
    flight_id: str
    flight_data: List[Dict[str, Any]]
    anomaly_report: Optional[Dict[str, Any]] = None
    flight_time: Optional[int] = None
    custom_prompt: Optional[str] = None  # User-provided classification instructions


class AnomalyClassification(BaseModel):
    """Structured response for anomaly classification."""
    rule_id: int = Field(description="The ID of the matched anomaly rule")
    rule_name: str = Field(description="The name of the matched anomaly rule")
    confidence: str = Field(description="Confidence level: High, Medium, or Low")
    reasoning: str = Field(description="Brief explanation of why this rule was chosen")


def parse_actions_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse JSON action blocks from the AI response (both code-fenced and plain JSON)."""
    actions = []
    
    # First, try to find JSON in code blocks
    json_block_pattern = r'```json\s*([\s\S]*?)```'
    matches = re.findall(json_block_pattern, response_text, re.IGNORECASE)
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and 'action' in parsed:
                actions.append(parsed)
        except json.JSONDecodeError:
            continue
    
    # Also look for plain JSON objects (not in code blocks)
    # Pattern: {"action": "...", ...}
    plain_json_pattern = r'\{["\']action["\']\s*:\s*["\'](?:highlight_point|highlight_segment|focus_time)["\'][^}]*\}'
    plain_matches = re.findall(plain_json_pattern, response_text)
    for match in plain_matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and 'action' in parsed:
                # Avoid duplicates (if already found in code block)
                if parsed not in actions:
                    actions.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return actions


def strip_actions_from_text(response_text: str) -> str:
    """Remove JSON action blocks from response text (both code-fenced and plain JSON)."""
    # Remove code-fenced JSON blocks
    cleaned = re.sub(r'```json\s*[\s\S]*?```', '', response_text, flags=re.IGNORECASE)
    
    # Remove plain JSON action objects
    cleaned = re.sub(r'\{["\']action["\']\s*:\s*["\'](?:highlight_point|highlight_segment|focus_time)["\'][^}]*\}', '', cleaned)
    
    # Clean up extra newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def extract_gemini_text(resp):
    """Extract text from Gemini response."""
    out = ""
    if hasattr(resp, 'candidates') and resp.candidates:
        for cand in resp.candidates:
            if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        out += part.text
    return out.strip()


def execute_reasoning_sql(query: str) -> Dict[str, Any]:
    """Execute SQL query against feedback_tagged.db for reasoning agent."""
    if not FEEDBACK_TAGGED_DB_PATH or not FEEDBACK_TAGGED_DB_PATH.exists():
        return {"error": "Feedback database not found", "rows": [], "count": 0}

    try:
        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        return {"error": str(e), "rows": [], "count": 0}


@router.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """
    Process a chat request using OpenAI Vision API.
    Generates a map image of the flight path and sends it to GPT for analysis.
    """
    try:
        # Get flight points
        points = request.points
        if not points:
            try:
                track_data = _get_unified_track(request.flight_id)
                points = track_data.get("points", [])
            except:
                points = []

        # Extract callsign
        callsign = None
        if points:
            for p in points:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        # Fetch flight details
        flight_details = _fetch_flight_details(request.flight_id, request.flight_time, callsign)

        # Generate map image
        map_image_base64 = None
        if points and len(points) >= 2:
            map_image_base64 = _generate_flight_map_image(points)

        # Build OpenAI messages
        openai_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        for msg in request.messages:
            if msg.get("role") in ["user", "assistant"]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build user content
        user_content = []

        if map_image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{map_image_base64}",
                    "detail": "high"
                }
            })

        # Add context
        flight_summary_text = _format_flight_summary_for_llm(flight_details, points)
        context_text = flight_summary_text + "\n\n"

        if request.analysis:
            summary = request.analysis.get("summary", {})
            layer1 = request.analysis.get("layer_1_rules", {})

            context_text += "=== ANOMALY ANALYSIS ===\n"
            if summary:
                context_text += f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}\n"
                context_text += f"Severity CNN: {summary.get('severity_cnn', 'N/A')}\n"
                context_text += f"Severity Dense: {summary.get('severity_dense', 'N/A')}\n"
                triggers = summary.get("triggers", [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_text += f"Triggers: {', '.join([str(t) for t in triggers])}\n"

            if layer1 and layer1.get("report", {}).get("matched_rules"):
                rules = layer1["report"]["matched_rules"]
                context_text += f"Matched Rules: {', '.join([r.get('name', str(r.get('id'))) for r in rules])}\n"

            context_text += "\n"

        context_text += f"User Question: {request.user_question}"

        user_content.append({"type": "text", "text": context_text})

        openai_messages.append({"role": "user", "content": user_content})

        # Call OpenAI
        response = _openai_client.chat.completions.create(
            model="gpt-5",
            messages=openai_messages
        )

        ai_response = response.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

gaza_actions = """פעילות הקואליציה הבין לאומית להצנחת סיוע הומניטרי בעזה לעתים נקרא ("ציפורי הטוב").במבצע השתתפו 7 מדינותירדן, איחוד האמירויות, מצרים, ארה"ב, צרפת, הולנד, בלגיה"""
saved_context = {"3d7211ef": """
טיסה חריגה מאוד ולא שגרתית, מטוס טס לישראל דרך לבנון באזור מאוים בשביל לחמוק ממזג אוויר בעייתי. סופת ביירון בתאריך זה התחוללה סופת ביירון שכללה מזג אוויר פעיל עם משקעים גבוהים שמשפיעה בצורה ישירה על נתיבי תעופה ותהליכי נחיתה, בין היתר בישראל, ירדן, לבנון, קפריסין ועוד.
""",
                 "3ad256ff": """ טיסה ככל הנראה צרפתית דיפלומטית (קיבלה קוד 3A שתואם לצרפת) לירדן עוקפת מצב ביטחוני ששורר בין ישראל לאיראן ומבצעת מעגלי המתנה לפני נחיתה."""
,
                 "3ad2166a":"""מטוס מנהלים ישראלי, ככל הנראה טיסה דיפלומטית, עוקף את המרחב האווירי של ישראל ככל הנראה בגלל המצב הבטחוני מול איראן ומבצע טיסה לעמאן."""
                 ,
                 "3adf2c74":"""באותו יום היה ירי טילים מאיראן לישראל במהלך מבצע עם כלביא נגד איראן הטיסות האלו ביצעו המתנות לפני נחיתה בעקבות ירי טילים מאיראן או מתימן לישראל"""
                 ,
                 "3ade9062":"""באותו יום היה ירי טילים מאיראן לישראל במהלך מבצע עם כלביא נגד איראן הטיסות האלו ביצעו המתנות לפני נחיתה בעקבות ירי טילים מאיראן או מתימן לישראל"""
                 ,"3af2006c":"""המטוס ביצע הנמכה חריגה והסטה חריגה מהמסלול טיסה שלו באמצע הדרך ליעד המקורי שלו בלבנון, המטוס ממשיך הנמכה ופונה לבסיס אחר הנמצא בערב הסעודית מה שמצביע על אירוע חירום"""
                 ,"3b5d3b75":"""טיסה צבאית אמריקאית שממריאה מבסיס חיל האוויר האמריקאי בקטאר.הטיסה עושה מסלול טיסה לקפריסין ואז לישראל ככל הנראה בגלל רגישות דיפלומטית, שלא יראה שהטיסה מגיעה ישירות לישראל ו"מבצעית מידי", היא מבצעת הנמכה ליד קפריסין, נעלמת מהרדאר ומופיעה שוב לכיוון ישראל בשביל לייצר הפרדה.שבפועל יראה כאילו הטיסה יצאה מקפריסין לישראל"""
                 ,"3b6519fd": """חליפה בין שני מטוסים"""
                 ,"3b85ce16":gaza_actions,
                 "3b85ca15":gaza_actions,
                 "3b85bab4":gaza_actions,
                 "3baedcf1":"""טיסת חירום של ארקיע שלא שינתה את הקוד לחירום אבל לפי פאטרן הטיסה שלה נראה שהיא ניסתה לבצע מספר נסיונות של גישה לנחיתה, מהלך להים ככל הנראה לטפל בתקלה ולנסות לנחות שוב, ככל הנראה מדובר על תקלה טכנית, יכול להיות בכני הנסע, מדפים וכדומה."""
                 ,"3bacace0":"""כטב"מ אמריקאי בטיסת איסוף מודיעין.לאור הפאטרן והסיווג של המטוס, נראה שהכטבם בנתיב טיסה של איסוף מודיעין לאורך חופי סוריה, לבנון וישראל."""
                 ,"3cb70c8b":"""טוס מקזחסטן לסוריה ואז מצרים מצביעה על טיסה דיפלומטית, המטוס רשום כמטוס גרמני. מבצע כמו מעין נחיתה זריזה ואז המראה או שהוא לא נחת, בשביל או לאסוף בכיר כלשהו למצרים או להראות כאילו הטיסה יצאה מסוריה"""
                 ,"3cb96e82": """המטוס הספציפי הזה הוא חדש לחלוטין (נמסר לחברה באוקטובר 2025, כ-10 ימים לפני הטיסה המוצגת). מטוסים חדשים מבצעים לעיתים קרובות "טיסות קבלה", טיסות הרצה או טיסות אימון לטייסים (Check flights). בטיסות אלו ממריאים, מבצעים תמרונים או בדיקות מערכות באזור מוגדר (במקרה זה דרומית לעמאן), וחוזרים לבסיס האם"""
                 ,"3cbe6a30":"""טיסה צבאית אמריקאית שבדרך כלל משמשת להטסות של בכירים ואישיויות חשובות. הטיסה המריאה וטיפסה לגובה גבוה לפני שיצאה לנתיב שלה לישראל בשביל לצבור גובה ולהתחמק מאיומים שקיימים באזור מכיוון תימן (החותים) והמסלול שהיא עשתה עוקף איומים, והתרחקות מיטבית מגבולות תימן דרך מדינות ערב שיש עימן שלום כמו סעודיה.יכול להיות שלטיסה הזו לא היה אישור לחצות מדינות באפריקה"""
                 ,"3cf959dd": """חזרה לנחיתה, נראה כמו תקלה במטוס, התקלה לא חמורה מכיוון שהמטוס לא ניגש ישר לנחיתה אלא עשה הרבה מאוד סיבובים וככל הנראה ניסה לתקן את התקלה באוויר, המטוס הנמיך לגובה 9 אלף רגל ושם הוא ביצע את רוב הסיבובים מה שיכול להצביע על תקלת דיחוס, ותפעול תקלה מתחת לגובה שמצריך דיחוס במטוס.בנוסף המטוס שהה באוויר זמן רב לפני נחיתה, יכול להצביע על כך שהוא היה עמוס בנוסעים ומטען והיה צריך "לבזבז" דלק בשביל לנחות"""
                 ,"3d0cd755":"""ככל הנראה כטב"ם צבאי,הטיסה מופיעה מעל דמשק בגובה 24 אלף רגל ומבצעת מעגלים הדוקים במשך כ-35 דקות המטוס שומר גובה ברדיוס קטן מאוד, מה שמצביע על אופי טיסה של כטב"ם צבאי. בדרך כלל כטבם צבאי מטפס לגובה במעגלים ואז מתחיל את משימתו. נתוני המהירות מראים מהירות יציבה של כ-140 קשר לאורך הטיסה התואם למהירות של כטבם, בנוסף אין מידע ונתונים על טיסה זו, מה שמצביעה על טיסה צבאית.המטוס מאבד מגע 22 קמ צפון מזרחית לבסיס חיל האוויר הירדני תוך כדי הנמכה ולא טס לכיוון עמאן (שדה תעופה אזרחי)"""
                 ,"3d1233db":"""הליכה סביב המטוס ניגש לביצוע נוסף של נחיתה"""
                 ,"3d1961da":"""מטוס אוסטריאן נוחת בקפריסין במקום ישראל ככל הנראה בגלל מזג אוויר"""
                 ,"3d1bb294":"""שינוי יעד נחיתה שהיה מתוכנן לביירות ככל הנראה בגלל מזג אוויר""",
                 "3d1aeecf":"""שינוי יעד נחיתה שהיה מתוכנן לביירות ככל הנראה בגלל מזג אוויר"""
                 ,"3d1fc7e7":"""מטוס ירדני טס לדמשק ביצע המתנות וחזר לנחיתה בעמאן. יכול להיות בגלל מזג אוויר, בטיחות טיסה או שדה סגור, המצב בסוריה נפיץ ויכול להיות שהשדה נסגר באותו רגע בגלל מצב בטחוני."""
                 ,"3d2e618c":"""ירדני אזרחי בטיסת תחזוקה או באימון טיסה """
                 ,"3d2e4bd9":"""שינוי יעד נחיתה"""
                 ,"3d4a26b3":"""טיסת כיול"""
                 ,"3d49bce2":"""טיסת כיול"""
                 ,"3d49a48e":"""טיסת כיול"""
                 ,"3d503012":"""מבנה צבאי ירדני"""
                 ,"3d64f7e0":"""מטוס תובלה אזרחי אמירתי חוצה לשטח ישראל בדרך לנחיתה בעמאן. וחורג מנתיב טיסה מוסכם.למטוסי נוסעים שלא מתוכננים לעבור בישראל (כמו בטיסה הזו שהם נוחתים בירדן) אין אישור לחצות קו גבול לישראל, המטוס סטה מהמסלול הרגיל המבצע גישה לנחיתה ולא במסלול סדיר המוסכם על המדינות והוא ביצע חריגה לשטח ישראל"""
                 ,"3d57e6bd":"""איבוד קליטה לא שגרתי אולי מצביעה על חסימות GPS במרחב"""
                 ,"3d6ed3b7":"""המטוס חורג מהנתיב טיסה המקובל לטיסה כזו וסוטה מהמסלול לטובת עקיפת מזג אוויר הוא נכנס מאזור חיפה וטס עד לנתב"ג סופת ביירון – בתאריך זה התחוללה סופת ביירון שכללה מזג אוויר פעיל עם משקעים גבוהים שמשפיעה בצורה ישירה על נתיבי תעופה ותהליכי נחיתה, בין היתר בישראל, ירדן, לבנון, קפריסין ועוד"""
                 ,"3d6e4e52":"""התקרבות מסוכנת"""
                 ,"3d72a082":"""טיסה רוסית אזרחית עוקפת את קפריסין כי ככל הנראה בגלל המלחמה מול אוקראינה אין לה אישור לחצות מעל מדינות האיחוד האירופי ולכן היא מבצעת עיקוף כזה"""
                 ,"3d7ff3e6":"""טיסה שהמריאה מקפריסין לירדן, הגיעה לירדן וחזרה לנחיתה בקפריסיןככל הנראה בגלל מזג אוויר או תקלה כלשהי בשדה התעופה בירדן, הטיסה חזרה כל הדרך לנחיתה בקפריסין ולא בבסיס קרוב יותר ככל הנראה כי בטיסות לואו קוסט אין לחברה בסיס לוגיסטי ויותר קל להם לטפל בלקוחות או במטוס בבסיס שבו יש צי לוגיסט"""
                 ,"3d7ee89a":"""טיסה שהלכה סביב נגעה בקרקע והמריאה שוב"""
                 ,"3d7f1bf8":"""טיסה שהמריאה מטורקיה טסה לירדן וחזרה על עקבותיה לטורקיה ונחתה לא באותו שדה שממנו המריאה.ככל הנראה בגלל מזג אוויר או תקלה כלשהי בשדה התעופה בירדן, הטיסה חזרה כל הדרך לנחיתה בטורקיה ולא בבסיס קרוב יותר ככל אין לחברה בסיס לוגיסטי ויותר קל להם לטפל בלקוחות או במטוס בבסיס שבו יש צי לוגיסטי."""
                 ,"3d8c3c09":"""טיסה של מטוס כנף ציון שככל הנראה משמשת לתחזוקה, אימון או העברה ולא לטיסה מבצעית של ראש הממשלה. ביצעה הליכה סביב"""
                 ,"3d92230e":"""ירדני שהגיע לנחות בעמאן, מנסה לנחות ולעשות נסיונות לנחיתה ומבטל נחיתה וטס לנחות בבסיס אחר"""
                 }

import time


@router.post("/api/ai/classify")
def ai_classify_anomaly_endpoint(request: ClassifyAnomalyRequest):
    """
    Classify an anomaly flight and choose the most appropriate anomaly reason from predefined rules.
    Uses Google Gemini with structured output to ensure consistent classification.
    """
    try:
        logger.info(f"AI Classify request for flight {request.flight_id}")

        # Extract callsign from flight data
        callsign = None
        if request.flight_data:
            for p in request.flight_data:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        flight_time = request.flight_time
        if not flight_time and request.flight_data:
            flight_time = request.flight_data[0].get('timestamp')

        # Load flight metadata from database instead of fetching
        try:
            from service.pg_provider import get_flight_metadata
            flight_details = get_flight_metadata(request.flight_id, schema='research')
            if not flight_details:
                # Fallback to basic info if not in DB
                flight_details = {
                    "flight_id": request.flight_id,
                    "callsign": callsign
                }
            else:
                # Ensure callsign is set
                if not flight_details.get('callsign') and callsign:
                    flight_details['callsign'] = callsign
        except Exception as e:
            logger.warning(f"Could not load metadata from DB: {e}, using basic info")
            flight_details = {
                "flight_id": request.flight_id,
                "callsign": callsign
            }

        # Build context
        context_parts = []
        flight_summary_text = _format_flight_summary_for_llm(flight_details, request.flight_data)
        context_parts.append(flight_summary_text)

        # Add time window
        if request.flight_data and len(request.flight_data) >= 2:
            ts0 = request.flight_data[0].get("timestamp")
            ts1 = request.flight_data[-1].get("timestamp")
            if ts0 and ts1:
                try:
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})")
                except Exception:
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0}\nEnd: {ts1}")

        # Generate map image
        map_image_base64 = None
        image_bytes = None
        mime_type = "image/png"
        
        if request.flight_data and len(request.flight_data) >= 2:
            map_image_base64 = _generate_flight_map_image(request.flight_data)
            if map_image_base64:
                try:
                    image_bytes = base64.b64decode(map_image_base64)
                    logger.info("Generated and decoded map image for classification")
                except Exception as e:
                    logger.error(f"Failed to decode generated map: {e}")


        if request.anomaly_report:
            context_parts.append("\n=== ANOMALY ANALYSIS ===")
            report = request.anomaly_report

            summary = report.get('summary', {})
            if summary:
                context_parts.append(f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}")
                context_parts.append(f"Confidence Score: {summary.get('confidence_score', 'N/A')}%")
                triggers = summary.get('triggers', [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_parts.append(f"Triggers: {', '.join(triggers)}")

            # Extract matched rules from layer_1_rules (matching data.json structure)
            layer1 = report.get('layer_1_rules', {})
            layer1_report = layer1.get('report', {})
            matched_rules = layer1_report.get('matched_rules', [])
            
            # Add matched rules information
            if matched_rules:
                context_parts.append("\n=== MATCHED RULES ===")
                for rule in matched_rules:
                    rule_name = rule.get('name', f"Rule {rule.get('id')}")
                    rule_summary = rule.get('summary', '')
                    if rule_summary:
                        context_parts.append(f"  - {rule_name}: {rule_summary}")
                    else:
                        context_parts.append(f"  - {rule_name}")
            
            # Check for proximity events in matched rules
            proximity_events = []
            for rule in matched_rules:
                # Check if this is a proximity rule (id 4 or name contains proximity/התקרבות)
                is_proximity_rule = rule.get('id') == 4 or 'proximity' in str(rule.get('name', '')).lower() or 'התקרבות' in str(rule.get('name', ''))
                if is_proximity_rule:
                    events = rule.get('details', {}).get('events', [])
                    if events:
                        for event in events:
                            if isinstance(event, dict):
                                proximity_events.append(event)
            
            # Add prominent proximity alert section if there are proximity events
            if proximity_events:
                context_parts.append("\n=== ⚠️ PROXIMITY ALERT - THIS FLIGHT HAS PROXIMITY EVENTS ===")
                
                # Generate summary statistics
                total_events = len(proximity_events)
                other_aircraft = set()
                distances = []
                alt_diffs = []
                
                for event in proximity_events:
                    callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    if callsign != 'Unknown':
                        other_aircraft.add(callsign)
                    if event.get('distance_nm') is not None:
                        try:
                            distances.append(float(event['distance_nm']))
                        except (ValueError, TypeError):
                            pass
                    if event.get('altitude_diff_ft') is not None:
                        try:
                            alt_diffs.append(float(event['altitude_diff_ft']))
                        except (ValueError, TypeError):
                            pass
                
                # Summary section
                context_parts.append("\nSUMMARY:")
                context_parts.append(f"  Total proximity events: {total_events}")
                context_parts.append(f"  Other aircraft involved: {', '.join(other_aircraft) if other_aircraft else 'Unknown'}")
                if distances:
                    context_parts.append(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} NM (min: {min(distances):.1f} NM)")
                if alt_diffs:
                    context_parts.append(f"  Altitude diff range: {min(alt_diffs):.0f} - {max(alt_diffs):.0f} ft")
                
                # Sample up to 5 events (evenly distributed if more than 5)
                if total_events <= 5:
                    sampled_events = proximity_events
                else:
                    # Sample evenly: first, last, and 3 from middle
                    indices = [0]
                    step = (total_events - 1) / 4
                    for i in range(1, 4):
                        indices.append(int(i * step))
                    indices.append(total_events - 1)
                    sampled_events = [proximity_events[i] for i in indices]
                
                context_parts.append(f"\nSAMPLED EVENTS ({len(sampled_events)} of {total_events}):")
                for i, event in enumerate(sampled_events, 1):
                    other_callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    other_flight_id = event.get('other_flight') or event.get('other_flight_id') or 'Unknown'
                    distance_nm = event.get('distance_nm', 'Unknown')
                    altitude_diff = event.get('altitude_diff_ft', 'Unknown')
                    timestamp = event.get('timestamp', 'Unknown')
                    
                    context_parts.append(f"  #{i}: {other_callsign} | Dist: {distance_nm} NM | Alt Diff: {altitude_diff} ft | TS: {timestamp}")
                
                context_parts.append("\nWhen answering questions, consider these proximity events and the other aircraft involved.")

        context_parts.append("\n=== TASK ===")
        if image_bytes:
            context_parts.append("A map visualization of the flight path is attached. Use it to analyze the flight pattern visually.")
        
        # Check if custom prompt is provided
        if request.custom_prompt:
            # Use custom prompt provided by user - free-form analysis
            context_parts.append("\n=== USER CLASSIFICATION INSTRUCTIONS ===")
            context_parts.append(request.custom_prompt)
            context_parts.append("\nBased on the flight data, map, and analysis above, provide your classification according to these instructions.")
            context_parts.append("Analyze what you observe and describe what happened without being constrained to predefined categories.")
            use_structured_output = False
        else:
            # Use predefined rules (backward compatibility)
            context_parts.append("Based on the flight data and map above, classify this anomaly by selecting the MOST PROBLEMATIC rule the flight violated from the list.")
            context_parts.append("Consider the flight pattern, triggers, matched rules, and any unusual behavior visible in the map.")
            context_parts.append("Provide your confidence level (High, Medium, or Low) and a brief reasoning for your choice.")
            use_structured_output = True

        full_prompt_text = "\n".join(context_parts)

        # Build Gemini request
        parts = [_types.Part(text=full_prompt_text)]
        
        # Add map image if available
        if image_bytes:
            parts.append(
                _types.Part(
                    inline_data=_types.Blob(
                        mime_type=mime_type,
                        data=image_bytes
                    )
                )
            )
            logger.info("Added map image to Gemini classification request")

        # Configure based on whether using structured output or custom prompt

        config = _types.GenerateContentConfig(
            system_instruction=request.custom_prompt,
                        tools=[_types.Tool(google_search=_types.GoogleSearch()), {'code_execution': {}}]

        )

        content = _types.Content(parts=parts)
        logger.info("Calling Gemini API for anomaly classification...")
        start_time = time.time()
        response = _gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            config=config,
            contents=[content]
        )
        logger.info(f"Gemini response time: {time.time() - start_time} seconds")
        
        # Extract response
        response_text = extract_gemini_text(response)
        
        # Return based on output type
        if use_structured_output:
            # Validate the response using Pydantic
            classification = AnomalyClassification.model_validate_json(response_text)
            
            # Find the full rule details
            matched_rule = next((r for r in RULES_METADATA if r['id'] == classification.rule_id), None)
            
            return {
                "classification": classification.model_dump(),
                "rule_details": matched_rule
            }
        else:
            # Return free-form classification based on custom prompt
            return {
                "classification": response_text,
                "custom_prompt": request.custom_prompt
            }

    except Exception as e:
        logger.error(f"AI Classify endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ai/classify_built_prompt")
def ai_classify_anomaly_endpoint_built_prompt(request: ClassifyAnomalyRequest):
    """
    Classify an anomaly flight and choose the most appropriate anomaly reason from predefined rules.
    Uses Google Gemini with structured output to ensure consistent classification.
    """
    try:
        logger.info(f"AI Classify request for flight {request.flight_id}")

        # Extract callsign from flight data
        callsign = None
        if request.flight_data:
            for p in request.flight_data:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        flight_time = request.flight_time
        if not flight_time and request.flight_data:
            flight_time = request.flight_data[0].get('timestamp')

        # Load flight metadata from database instead of fetching
        try:
            from service.pg_provider import get_flight_metadata
            flight_details = get_flight_metadata(request.flight_id, schema='research')
            if not flight_details:
                # Fallback to basic info if not in DB
                flight_details = {
                    "flight_id": request.flight_id,
                    "callsign": callsign
                }
            else:
                # Ensure callsign is set
                if not flight_details.get('callsign') and callsign:
                    flight_details['callsign'] = callsign
        except Exception as e:
            logger.warning(f"Could not load metadata from DB: {e}, using basic info")
            flight_details = {
                "flight_id": request.flight_id,
                "callsign": callsign
            }

        # Build context
        context_parts = []
        flight_summary_text = _format_flight_summary_for_llm(flight_details, request.flight_data)
        context_parts.append(flight_summary_text)

        # Add time window
        if request.flight_data and len(request.flight_data) >= 2:
            ts0 = request.flight_data[0].get("timestamp")
            ts1 = request.flight_data[-1].get("timestamp")
            if ts0 and ts1:
                try:
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})")
                except Exception:
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0}\nEnd: {ts1}")

        # Generate map image
        map_image_base64 = None
        image_bytes = None
        mime_type = "image/png"
        
        if request.flight_data and len(request.flight_data) >= 2:
            map_image_base64 = _generate_flight_map_image(request.flight_data)
            if map_image_base64:
                try:
                    image_bytes = base64.b64decode(map_image_base64)
                    logger.info("Generated and decoded map image for classification")
                except Exception as e:
                    logger.error(f"Failed to decode generated map: {e}")


        if request.anomaly_report:
            context_parts.append("\n=== ANOMALY ANALYSIS ===")
            report = request.anomaly_report

            summary = report.get('summary', {})
            if summary:
                context_parts.append(f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}")
                context_parts.append(f"Confidence Score: {summary.get('confidence_score', 'N/A')}%")
                triggers = summary.get('triggers', [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_parts.append(f"Triggers: {', '.join(triggers)}")

            # Extract matched rules from layer_1_rules (matching data.json structure)
            layer1 = report.get('layer_1_rules', {})
            layer1_report = layer1.get('report', {})
            matched_rules = layer1_report.get('matched_rules', [])
            
            # Add matched rules information
            if matched_rules:
                context_parts.append("\n=== MATCHED RULES ===")
                for rule in matched_rules:
                    rule_name = rule.get('name', f"Rule {rule.get('id')}")
                    rule_summary = rule.get('summary', '')
                    if rule_summary:
                        context_parts.append(f"  - {rule_name}: {rule_summary}")
                    else:
                        context_parts.append(f"  - {rule_name}")
            
            # Check for proximity events in matched rules
            proximity_events = []
            for rule in matched_rules:
                # Check if this is a proximity rule (id 4 or name contains proximity/התקרבות)
                is_proximity_rule = rule.get('id') == 4 or 'proximity' in str(rule.get('name', '')).lower() or 'התקרבות' in str(rule.get('name', ''))
                if is_proximity_rule:
                    events = rule.get('details', {}).get('events', [])
                    if events:
                        for event in events:
                            if isinstance(event, dict):
                                proximity_events.append(event)
            
            # Add prominent proximity alert section if there are proximity events
            if proximity_events:
                context_parts.append("\n=== ⚠️ PROXIMITY ALERT - THIS FLIGHT HAS PROXIMITY EVENTS ===")
                
                # Generate summary statistics
                total_events = len(proximity_events)
                other_aircraft = set()
                distances = []
                alt_diffs = []
                
                for event in proximity_events:
                    callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    if callsign != 'Unknown':
                        other_aircraft.add(callsign)
                    if event.get('distance_nm') is not None:
                        try:
                            distances.append(float(event['distance_nm']))
                        except (ValueError, TypeError):
                            pass
                    if event.get('altitude_diff_ft') is not None:
                        try:
                            alt_diffs.append(float(event['altitude_diff_ft']))
                        except (ValueError, TypeError):
                            pass
                
                # Summary section
                context_parts.append("\nSUMMARY:")
                context_parts.append(f"  Total proximity events: {total_events}")
                context_parts.append(f"  Other aircraft involved: {', '.join(other_aircraft) if other_aircraft else 'Unknown'}")
                if distances:
                    context_parts.append(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} NM (min: {min(distances):.1f} NM)")
                if alt_diffs:
                    context_parts.append(f"  Altitude diff range: {min(alt_diffs):.0f} - {max(alt_diffs):.0f} ft")
                
                # Sample up to 5 events (evenly distributed if more than 5)
                if total_events <= 5:
                    sampled_events = proximity_events
                else:
                    # Sample evenly: first, last, and 3 from middle
                    indices = [0]
                    step = (total_events - 1) / 4
                    for i in range(1, 4):
                        indices.append(int(i * step))
                    indices.append(total_events - 1)
                    sampled_events = [proximity_events[i] for i in indices]
                
                context_parts.append(f"\nSAMPLED EVENTS ({len(sampled_events)} of {total_events}):")
                for i, event in enumerate(sampled_events, 1):
                    other_callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    other_flight_id = event.get('other_flight') or event.get('other_flight_id') or 'Unknown'
                    distance_nm = event.get('distance_nm', 'Unknown')
                    altitude_diff = event.get('altitude_diff_ft', 'Unknown')
                    timestamp = event.get('timestamp', 'Unknown')
                    
                    context_parts.append(f"  #{i}: {other_callsign} | Dist: {distance_nm} NM | Alt Diff: {altitude_diff} ft | TS: {timestamp}")
                
                context_parts.append("\nWhen answering questions, consider these proximity events and the other aircraft involved.")

        context_parts.append("\n=== TASK ===")
        if image_bytes:
            context_parts.append("A map visualization of the flight path is attached. Use it to analyze the flight pattern visually.")
        
        # Check if custom prompt is provided
        if request.custom_prompt:
            # Use custom prompt provided by user - free-form analysis
            context_parts.append("\n=== USER CLASSIFICATION INSTRUCTIONS ===")
            context_parts.append(request.custom_prompt)
            context_parts.append("\nBased on the flight data, map, and analysis above, provide your classification according to these instructions.")
            context_parts.append("Analyze what you observe and describe what happened without being constrained to predefined categories.")
            use_structured_output = False
        else:
            # Use predefined rules (backward compatibility)
            context_parts.append("Based on the flight data and map above, classify this anomaly by selecting the MOST PROBLEMATIC rule the flight violated from the list.")
            context_parts.append("Consider the flight pattern, triggers, matched rules, and any unusual behavior visible in the map.")
            context_parts.append("Provide your confidence level (High, Medium, or Low) and a brief reasoning for your choice.")
            use_structured_output = True

        full_prompt_text = "\n".join(context_parts)

        # Build Gemini request
        parts = [_types.Part(text=full_prompt_text)]
        
        # Add map image if available
        if image_bytes:
            parts.append(
                _types.Part(
                    inline_data=_types.Blob(
                        mime_type=mime_type,
                        data=image_bytes
                    )
                )
            )
            logger.info("Added map image to Gemini classification request")

        # Configure based on whether using structured output or custom prompt

        config = _types.GenerateContentConfig(
            system_instruction="""As an expert aviation data analyst, your core mission is to perform a surgical inference of the root cause by correlating the detected flight anomaly with the provided environmental context. You must move beyond simple observation to determine exactly why the anomaly was a logical necessity or a specific response to the surrounding conditions, ensuring that the environmental data justifies the flight behavior. It is critical that your final output is restricted to a professional summary of exactly three to six words, providing only the ultimate root cause without any introductory phrases, filler text, or repetition of the input """,
                        tools=[_types.Tool(google_search=_types.GoogleSearch()), {'code_execution': {}}]

        )

        content = _types.Content(parts=parts)
        logger.info("Calling Gemini API for anomaly classification...")
        start_time = time.time()
        response = _gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            config=config,
            contents=[content]
        )
        logger.info(f"Gemini response time: {time.time() - start_time} seconds")
        
        # Extract response
        response_text = extract_gemini_text(response)
        
        # Return based on output type
        if use_structured_output:
            # Validate the response using Pydantic
            classification = AnomalyClassification.model_validate_json(response_text)
            
            # Find the full rule details
            matched_rule = next((r for r in RULES_METADATA if r['id'] == classification.rule_id), None)
            
            return {
                "classification": classification.model_dump(),
                "rule_details": matched_rule
            }
        else:
            # Return free-form classification based on custom prompt
            return {
                "classification": response_text,
                "custom_prompt": request.custom_prompt
            }

    except Exception as e:
        logger.error(f"AI Classify endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/api/ai/analyze")
def ai_analyze_endpoint(request: AIAnalyzeRequest):
    """
    AI Co-Pilot endpoint that analyzes a flight with screenshot support.
    Uses Google Gemini for analysis.
    """
    try:
        logger.info(f"AI Analyze request for flight {request.flight_id}, history: {len(request.history)} messages")

        # Prepare image data
        image_bytes = None
        mime_type = "image/png"

        if request.screenshot:
            screenshot_data = request.screenshot
            if screenshot_data.startswith('data:'):
                try:
                    header, encoded = screenshot_data.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    image_bytes = base64.b64decode(encoded)
                except Exception as e:
                    logger.error(f"Failed to parse data URL: {e}")
                    try:
                        image_bytes = base64.b64decode(screenshot_data)
                    except:
                        pass
            else:
                try:
                    image_bytes = base64.b64decode(screenshot_data)
                except Exception as e:
                    logger.error(f"Failed to decode base64 screenshot: {e}")

        if not image_bytes and request.flight_data and len(request.flight_data) >= 2:
            map_image_base64 = _generate_flight_map_image(request.flight_data)
            if map_image_base64:
                try:
                    image_bytes = base64.b64decode(map_image_base64)
                    mime_type = "image/png"
                except Exception as e:
                    logger.error(f"Failed to decode generated map: {e}")

        # Extract callsign
        callsign = None
        if request.flight_data:
            for p in request.flight_data:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

        flight_time = request.flight_time
        if not flight_time and request.flight_data:
            flight_time = request.flight_data[0].get('timestamp')

        # Fetch flight details
        flight_details = _fetch_flight_details(request.flight_id, flight_time, callsign) if flight_time else {
            "flight_id": request.flight_id,
            "callsign": callsign
        }

        # Build context
        context_parts = []
        flight_summary_text = _format_flight_summary_for_llm(flight_details, request.flight_data)
        context_parts.append(flight_summary_text)

        # Add time window
        if request.flight_data and len(request.flight_data) >= 2:
            ts0 = request.flight_data[0].get("timestamp")
            ts1 = request.flight_data[-1].get("timestamp")
            if ts0 and ts1:
                try:
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})")
                except Exception:
                    context_parts.append(f"\n=== TIME RANGE ===\nStart: {ts0}\nEnd: {ts1}")
        
        # Track proximity aircraft for map generation
        proximity_flight_ids = set()
        
        if request.anomaly_report:
            context_parts.append("\n=== ANOMALY ANALYSIS ===")
            report = request.anomaly_report

            summary = report.get('summary', {})
            if summary:
                context_parts.append(f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}")
                context_parts.append(f"Confidence Score: {summary.get('confidence_score', 'N/A')}%")
                triggers = summary.get('triggers', [])
                triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                if triggers:
                    context_parts.append(f"Triggers: {', '.join(triggers)}")

            # Check for proximity events and extract other flight information
            proximity_events = []
            layer1 = report.get('layer_1_rules', {})
            
            # Also check for matched_rules at top level of full_report
            all_rules_sources = [
                layer1.get('report', {}).get('matched_rules', []),
                report.get('matched_rules', []),
                report.get('full_report', {}).get('matched_rules', []) if isinstance(report.get('full_report'), dict) else [],
            ]
            
            for rules in all_rules_sources:
                if not rules:
                    continue
                for rule in rules:
                    # Check if this is a proximity rule (id 4 or name contains proximity)
                    is_proximity_rule = rule.get('id') == 4 or 'proximity' in str(rule.get('name', '')).lower()
                    if is_proximity_rule and rule.get('details', {}).get('events'):
                        for event in rule['details']['events']:
                            if isinstance(event, dict):
                                proximity_events.append(event)
            
            # Add prominent proximity alert section if there are proximity events
            if proximity_events:
                context_parts.append("\n=== ⚠️ PROXIMITY ALERT - THIS FLIGHT HAS PROXIMITY EVENTS ===")
                
                # Generate summary statistics
                total_events = len(proximity_events)
                other_aircraft = set()
                distances = []
                alt_diffs = []
                
                for event in proximity_events:
                    callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    if callsign != 'Unknown':
                        other_aircraft.add(callsign)
                    
                    # Collect flight IDs for map generation
                    other_flight_id = event.get('other_flight') or event.get('other_flight_id')
                    if other_flight_id and other_flight_id != 'Unknown':
                        proximity_flight_ids.add(other_flight_id)
                    
                    if event.get('distance_nm') is not None:
                        try:
                            distances.append(float(event['distance_nm']))
                        except (ValueError, TypeError):
                            pass
                    if event.get('altitude_diff_ft') is not None:
                        try:
                            alt_diffs.append(float(event['altitude_diff_ft']))
                        except (ValueError, TypeError):
                            pass
                
                # Summary section
                context_parts.append("\nSUMMARY:")
                context_parts.append(f"  Total proximity events: {total_events}")
                context_parts.append(f"  Other aircraft involved: {', '.join(other_aircraft) if other_aircraft else 'Unknown'}")
                if distances:
                    context_parts.append(f"  Distance range: {min(distances):.1f} - {max(distances):.1f} NM (min: {min(distances):.1f} NM)")
                if alt_diffs:
                    context_parts.append(f"  Altitude diff range: {min(alt_diffs):.0f} - {max(alt_diffs):.0f} ft")
                
                # Sample up to 5 events (evenly distributed if more than 5)
                if total_events <= 5:
                    sampled_events = proximity_events
                else:
                    # Sample evenly: first, last, and 3 from middle
                    indices = [0]
                    step = (total_events - 1) / 4
                    for i in range(1, 4):
                        indices.append(int(i * step))
                    indices.append(total_events - 1)
                    sampled_events = [proximity_events[i] for i in indices]
                
                context_parts.append(f"\nSAMPLED EVENTS ({len(sampled_events)} of {total_events}):")
                for i, event in enumerate(sampled_events, 1):
                    other_callsign = event.get('other_callsign') or event.get('other_flight') or 'Unknown'
                    other_flight_id = event.get('other_flight') or event.get('other_flight_id') or 'Unknown'
                    distance_nm = event.get('distance_nm', 'Unknown')
                    altitude_diff = event.get('altitude_diff_ft', 'Unknown')
                    timestamp = event.get('timestamp', 'Unknown')
                    
                    context_parts.append(f"  #{i}: {other_callsign} | Dist: {distance_nm} NM | Alt Diff: {altitude_diff} ft | TS: {timestamp}")
                
                context_parts.append("\nWhen answering questions, consider these proximity events and the other aircraft involved.")
        
        # Also check for proximity_context provided directly from frontend
        if request.proximity_context and len(request.proximity_context) > 0:
            # Filter out events already covered from anomaly_report
            new_prox_events = []
            for prox in request.proximity_context:
                # Collect flight IDs for map generation
                if prox.other_flight_id and prox.other_flight_id != 'Unknown':
                    proximity_flight_ids.add(prox.other_flight_id)
                
                if proximity_events:
                    already_covered = any(
                        e.get('other_flight') == prox.other_flight_id or 
                        e.get('other_callsign') == prox.other_callsign
                        for e in proximity_events
                    )
                    if already_covered:
                        continue
                new_prox_events.append(prox)
            
            if new_prox_events:
                # Only add header if we didn't already add from anomaly_report
                if not proximity_events:
                    context_parts.append("\n=== ⚠️ PROXIMITY ALERT - THIS FLIGHT HAS PROXIMITY EVENTS ===")
                    
                    # Generate summary for frontend-provided context
                    total_events = len(new_prox_events)
                    other_aircraft = set()
                    for prox in new_prox_events:
                        if prox.other_callsign:
                            other_aircraft.add(prox.other_callsign)
                    
                    context_parts.append("\nSUMMARY:")
                    context_parts.append(f"  Total proximity events: {total_events}")
                    context_parts.append(f"  Other aircraft involved: {', '.join(other_aircraft) if other_aircraft else 'Unknown'}")
                
                # Sample up to 5 events
                if len(new_prox_events) <= 5:
                    sampled_prox = new_prox_events
                else:
                    indices = [0]
                    step = (len(new_prox_events) - 1) / 4
                    for i in range(1, 4):
                        indices.append(int(i * step))
                    indices.append(len(new_prox_events) - 1)
                    sampled_prox = [new_prox_events[i] for i in indices]
                
                context_parts.append(f"\npresenting ({len(sampled_prox)} events out of {len(new_prox_events)}):")
                for i, prox in enumerate(sampled_prox, 1):
                    parts = []
                    if prox.other_callsign:
                        parts.append(prox.other_callsign)
                    elif prox.other_flight_id:
                        parts.append(prox.other_flight_id)
                    if prox.distance_nm is not None:
                        parts.append(f"Dist: {prox.distance_nm} nautical  mile")
                    if prox.altitude_diff_ft is not None:
                        parts.append(f"Alt Diff: {prox.altitude_diff_ft} ft")
                    if prox.timestamp is not None:
                        parts.append(f"TS: {prox.timestamp}")
                    context_parts.append(f"  #{i}: {' | '.join(parts)}")

            if layer1:
                rules = layer1.get('report', {}).get('matched_rules', [])
                if rules:
                    context_parts.append("\nMatched Rules:")
                    for rule in rules:
                        rule_name = rule.get('name', f"Rule {rule.get('id')}")
                        context_parts.append(f"  - {rule_name}")
                        if rule.get('details'):
                            details = rule['details']
                            if 'events' in details:
                                event_strs = []
                                for e in details['events']:
                                    if isinstance(e, dict):
                                        summary_str = "; ".join(f"{k}: {v}" for k, v in e.items())
                                        event_strs.append(f"{{{summary_str}}}")
                                    else:
                                        event_strs.append(str(e))
                                context_parts.append(f"    Events: {', '.join(event_strs)}")

        context_parts.append(f"\n=== USER QUESTION ===\n{request.question}")
        additional_data = saved_context.get(request.flight_id)
        if additional_data:
            context_parts.append(f"\n system summery\n{additional_data}")

        full_prompt_text = "\n".join(context_parts)

        system_instruction_reasoning = (
            "Give a good comprehensive answer. "
            # "Keep in mind the system can be wrong."
            "Allways answer in hebrew."
            "The provided image is a cutoff for the system boundery box that includes the flight path over a map, use it to understand the flight path and the anomalies"
            "The system summery is llm that reasoned about the data and made research, trust it and build the answer around it" if additional_data else ""
            "Always include 2–4 plausible explanations or details the user might not have considered.\n\n"
            # "## MAP HIGHLIGHTING ACTIONS\n"
            # "When you want to point to something on the map, output a JSON action block.\n\n"
            # "Available Actions:\n"
            # '1. Highlight a specific point: `{"action": "highlight_point", "lat": 32.1234, "lon": 34.9876}`\n'
            # '2. Highlight a segment: `{"action": "highlight_segment", "startIndex": 120, "endIndex": 150}`\n\n'
            # "Use AT MOST one or two actions per response."
        )

        system_instruction_fast = (
            # "Keep in mind the system can be wrong."
            "Allways answer in hebrew."
            "Answer shortly and to the point."
            # "Available Actions:\n"
            # '1. Highlight a specific point: `{"action": "highlight_point", "lat": 32.1234, "lon": 34.9876}`\n'
            # '2. Highlight a segment: `{"action": "highlight_segment", "startIndex": 120, "endIndex": 150}`\n\n'
            # "Use AT MOST one or two actions per response."
        )
        
        # Generate flight maps for proximity aircraft
        proximity_maps = {}
        if proximity_flight_ids:
            logger.info(f"Generating maps for {len(proximity_flight_ids)} proximity aircraft: {proximity_flight_ids}")
            for other_flight_id in proximity_flight_ids:
                try:
                    # Fetch track data for the proximity aircraft
                    track_data = _get_unified_track(other_flight_id)
                    points = track_data.get("points", []) if track_data else []
                    
                    if points and len(points) >= 2:
                        # Generate map image
                        map_base64 = _generate_flight_map_image(points)
                        if map_base64:
                            proximity_maps[other_flight_id] = map_base64
                            logger.info(f"Generated map for proximity aircraft {other_flight_id}")
                        else:
                            logger.warning(f"Failed to generate map for proximity aircraft {other_flight_id}")
                    else:
                        logger.warning(f"Insufficient track points for proximity aircraft {other_flight_id}: {len(points) if points else 0} points")
                except Exception as e:
                    logger.error(f"Error generating map for proximity aircraft {other_flight_id}: {e}")
        
        # Build Gemini request
        parts = [_types.Part(text=full_prompt_text)]

        if image_bytes:
            parts.append(
                _types.Part(
                    inline_data=_types.Blob(
                        mime_type=mime_type,
                        data=image_bytes
                    )
                )
            )
        
        # Add proximity aircraft maps to Gemini request
        if proximity_maps:
            logger.info(f"Adding {len(proximity_maps)} proximity aircraft maps to Gemini request")
            for flight_id, map_base64 in proximity_maps.items():
                try:
                    map_bytes = base64.b64decode(map_base64)
                    parts.append(
                        _types.Part(
                            inline_data=_types.Blob(
                                mime_type="image/png",
                                data=map_bytes
                            )
                        )
                    )
                    # Add a text part labeling this image
                    parts.insert(-1, _types.Part(text=f"\n[Map of proximity aircraft: {flight_id}]"))
                    logger.info(f"Added map image for proximity aircraft {flight_id}")
                except Exception as e:
                    logger.error(f"Failed to add map for proximity aircraft {flight_id}: {e}")
                    
        tools = []
        if request.mode != "on-prem":
            tools.append(_types.Tool(google_search=_types.GoogleSearch()))
            tools.append({'code_execution': {}})
        
        system_instruction = system_instruction_fast if request.mode in ["fast", "on-prem"] else system_instruction_reasoning
        full_system_instruction = system_instruction
        if request.mode != "on-prem":
            full_system_instruction += """ CRITICAL INSTRUCTION: 
            1. For any calculations, ALWAYS use the code execution tool. 
            2. Do not perform math 'in-head'. Write Python code to solve the formula and present the result. """
        config = _types.GenerateContentConfig(
            system_instruction=full_system_instruction ,
            tools=tools
        )

        content = _types.Content(parts=parts)
        # Select model based on mode: fast uses gemini-2.5-flash, thinking uses gemini-3-pro-preview
        model_name = "gemini-3-flash-preview" if request.mode in ["fast", "on-prem"] else "gemini-3-pro-preview"
        logger.info(f"Calling Gemini API with model: {model_name} (mode: {request.mode})...")
        start_time = time.time()
        response = _gemini_client.models.generate_content(
            model=model_name,
            config=config,
            contents=[content]
        )
        logger.info(f"Gemini response time: {time.time() - start_time} seconds")
        
        ai_response = extract_gemini_text(response)
        

        actions = parse_actions_from_response(ai_response)
        clean_response = strip_actions_from_text(ai_response) if actions else ai_response

        return {
            "response": clean_response,
            "actions": actions
        }

    except Exception as e:
        logger.error(f"AI Analyze endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


REASONING_AGENT_PROMPT = """
You are an aviation assistant. Answer questions using the context provided.

## SQL TOOL
You have access to a SQL database (feedback_tagged.db) with user-tagged flights. To query it:
<tool>sql: YOUR_QUERY</tool>

## DATABASE SCHEMA

### flight_metadata (flight details)
- flight_id (TEXT, PK), callsign, flight_number, airline, airline_code
- aircraft_type, aircraft_model, aircraft_registration
- origin_airport, destination_airport
- first_seen_ts, last_seen_ts (Unix timestamps)
- flight_duration_sec, total_points
- min/max/avg_altitude_ft, min/max/avg_speed_kts
- is_military, military_type, emergency_squawk_detected

### anomaly_reports (analysis results)
- flight_id (TEXT, PK), timestamp, is_anomaly
- severity_cnn, severity_dense (0-100 scores)
- matched_rule_ids, matched_rule_names (comma-separated)
- full_report (JSON)

### user_feedback (tagging info)
- flight_id (TEXT, PK), tagged_at, user_label (1=anomaly, 0=normal)
- rule_id, rule_name, comments, other_details

### flight_tracks (track points)
- flight_id, timestamp, lat, lon, alt, gspeed, track, squawk, callsign

## FETCH AND RETURN
To show flights to the user, wrap your SQL in <fetch and return> tags:
<fetch and return>SELECT flight_id, timestamp, is_anomaly, severity_cnn, severity_dense, full_report FROM anomaly_reports WHERE ...</fetch and return>

Be concise and helpful. If asked about specific flights, use SQL to find them.
"""


def run_reasoning_agent(
        user_message: str,
        conversation_history: List[Dict[str, str]] = None,
        max_steps: int = 5,
        map_image_base64: str = None,
        flight_context: str = None
) -> Dict[str, Any]:
    """Run the reasoning agent with tool use capabilities."""

    prompt_parts = [REASONING_AGENT_PROMPT]

    if flight_context:
        prompt_parts.append(f"\n--- CURRENT FLIGHT CONTEXT ---\n{flight_context}")

    if conversation_history:
        prompt_parts.append("\n--- CONVERSATION HISTORY ---")
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

    prompt_parts.append(f"\n--- USER MESSAGE ---\n{user_message}")

    full_prompt = "\n".join(prompt_parts)

    parts = [_types.Part(text=full_prompt)]

    if map_image_base64:
        try:
            image_bytes = base64.b64decode(map_image_base64)
            parts.append(
                _types.Part(
                    inline_data=_types.Blob(
                        mime_type="image/png",
                        data=image_bytes
                    )
                )
            )
            logger.info("[REASONING AGENT] Added map image to Gemini request")
        except Exception as e:
            logger.error(f"[REASONING AGENT] Failed to decode image: {e}")

    config = _types.GenerateContentConfig(
        system_instruction="You are a helpful aviation assistant. Be concise.\n\n"
                           + """ CRITICAL INSTRUCTION: 
1. For any calculations, ALWAYS use the code execution tool. 
2. Do not perform math 'in-head'. Write Python code to solve the formula and present the result. """,
        tools=[_types.Tool(google_search=_types.GoogleSearch()), {'code_execution': {}}]
    )

    for step in range(max_steps):
        logger.info(f"[REASONING AGENT] Step {step + 1}/{max_steps}")

        try:
            content = _types.Content(parts=parts)
            response = _gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=[content]
            )
            msg = extract_gemini_text(response)
            logger.info(f"[REASONING AGENT] Gemini output:\n{msg[:500]}...")
        except Exception as e:
            logger.error(f"[REASONING AGENT] Gemini error: {e}")
            return {"type": "message", "response": f"Error calling AI: {e}"}

        # Check for <fetch and return>
        if "<fetch and return>" in msg and "</fetch and return>" in msg:
            try:
                sql_query = msg.split("<fetch and return>")[1].split("</fetch and return>")[0].strip()
                logger.info(f"[REASONING AGENT] Fetch and return SQL: {sql_query}")

                result = execute_reasoning_sql(sql_query)

                if "error" in result:
                    return {"type": "message", "response": f"SQL error: {result['error']}"}

                flights = []
                raw_rows = result.get("rows", [])
                flight_ids = [r.get("flight_id") for r in raw_rows if r.get("flight_id")]
                callsigns = {}

                if flight_ids and FEEDBACK_TAGGED_DB_PATH.exists():
                    try:
                        conn = sqlite3.connect(str(FEEDBACK_TAGGED_DB_PATH))
                        cursor = conn.cursor()
                        placeholders = ",".join(["?"] * len(flight_ids))
                        try:
                            cursor.execute(
                                f"SELECT flight_id, callsign FROM flight_metadata WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL AND callsign != ''",
                                flight_ids)
                            for fid, cs in cursor.fetchall():
                                if cs: callsigns[fid] = cs
                        except:
                            pass
                        try:
                            cursor.execute(
                                f"SELECT DISTINCT flight_id, callsign FROM flight_tracks WHERE flight_id IN ({placeholders}) AND callsign IS NOT NULL AND callsign != ''",
                                flight_ids)
                            for fid, cs in cursor.fetchall():
                                if cs and fid not in callsigns: callsigns[fid] = cs
                        except:
                            pass
                        conn.close()
                    except Exception as e:
                        logger.error(f"Error fetching callsigns: {e}")

                for row in raw_rows:
                    report = row.get("full_report")
                    if isinstance(report, str):
                        try:
                            report = json.loads(report)
                        except:
                            report = {}

                    flight_id = row.get("flight_id")
                    callsign = callsigns.get(flight_id)
                    if not callsign and isinstance(report, dict):
                        callsign = report.get("summary", {}).get("callsign")

                    flights.append({
                        "flight_id": flight_id,
                        "timestamp": row.get("timestamp"),
                        "is_anomaly": bool(row.get("is_anomaly")),
                        "severity_cnn": row.get("severity_cnn"),
                        "severity_dense": row.get("severity_dense"),
                        "full_report": report,
                        "callsign": callsign
                    })

                response_text = msg.split("<fetch and return>")[0].strip()
                if not response_text:
                    response_text = f"Found {len(flights)} flight(s)."

                return {
                    "type": "flights",
                    "response": response_text,
                    "flights": flights
                }

            except Exception as e:
                logger.error(f"[REASONING AGENT] Fetch error: {e}")
                return {"type": "message", "response": f"Error: {e}"}

        # Check for SQL tool
        if "<tool>" in msg and "sql:" in msg.lower():
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
                sql_query = tool_block.split("sql:", 1)[1].strip()

                logger.info(f"[REASONING AGENT] SQL tool: {sql_query}")
                sql_result = execute_reasoning_sql(sql_query)

                result_str = json.dumps(sql_result, indent=2, default=str)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + "\n... (truncated)"

                parts = [_types.Part(
                    text=full_prompt + f"\n\nTool result (sql):\n{result_str}\n\nNow answer the user based on this data.")]
                if map_image_base64:
                    try:
                        image_bytes = base64.b64decode(map_image_base64)
                        parts.append(_types.Part(inline_data=_types.Blob(mime_type="image/png", data=image_bytes)))
                    except:
                        pass
                continue
            except Exception as e:
                logger.error(f"[REASONING AGENT] SQL error: {e}")
                parts = [_types.Part(text=full_prompt + f"\n\nSQL error: {e}\n\nTry a different approach.")]
                continue

        # No tools - return response
        logger.info("[REASONING AGENT] Returning response")
        return {"type": "message", "response": msg.strip()}

    logger.warning("[REASONING AGENT] Max steps reached")
    return {"type": "message", "response": "Could not complete the request. Please try again."}


@router.post("/api/ai/reasoning")
def reasoning_endpoint(request: ReasoningRequest):
    """
    AI Reasoning Agent endpoint.
    Accepts a user message and conversation history.
    Optionally accepts flight context for visual analysis.
    """
    try:
        logger.info(f"[REASONING API] Message: {request.message[:100]}...")

        map_image_base64 = None
        flight_context = None

        if request.flight_id and request.points and len(request.points) >= 2:
            logger.info(f"[REASONING API] Generating map for flight {request.flight_id}")
            map_image_base64 = _generate_flight_map_image(request.points)

            callsign = None
            for p in request.points:
                if p.get("callsign"):
                    callsign = p["callsign"]
                    break

            flight_time = request.points[0].get("timestamp", 0) if request.points else 0

            logger.info(f"[REASONING API] Fetching flight details for {request.flight_id}...")
            flight_details = _fetch_flight_details(request.flight_id, flight_time, callsign)

            flight_context = _format_flight_summary_for_llm(flight_details, request.points)
            flight_context += "\n"

            try:
                ts0 = request.points[0].get("timestamp")
                ts1 = request.points[-1].get("timestamp")
                if ts0 and ts1:
                    iso0 = datetime.fromtimestamp(int(ts0), tz=timezone.utc).isoformat()
                    iso1 = datetime.fromtimestamp(int(ts1), tz=timezone.utc).isoformat()
                    flight_context += f"\n=== TIME RANGE ===\nStart: {ts0} ({iso0})\nEnd: {ts1} ({iso1})\n"
            except Exception:
                pass

            if request.anomaly_report:
                summary = request.anomaly_report.get("summary", {})
                layer1 = request.anomaly_report.get("layer_1_rules", {})

                flight_context += "\n=== ANOMALY ANALYSIS ===\n"
                if summary:
                    flight_context += f"Is Anomaly: {summary.get('is_anomaly', 'Unknown')}\n"
                    flight_context += f"Confidence Score: {summary.get('confidence_score', 'N/A')}%\n"
                    triggers = summary.get('triggers', [])
                    triggers = _rewrite_triggers_with_feedback(triggers, request.flight_id)
                    if triggers:
                        flight_context += f"Triggers: {', '.join([str(t) for t in triggers])}\n"

                if layer1 and layer1.get("report", {}).get("matched_rules"):
                    rules = layer1["report"]["matched_rules"]
                    flight_context += f"Matched Rules: {', '.join([r.get('name', str(r.get('id'))) for r in rules])}\n"

            flight_context += "\nA map image of this flight's path is attached.\n"

        if flight_context:
            prefixed_context = (
                    "You are looking at ONE SPECIFIC FLIGHT. The score and anomaly decision the system created can be wrong, so double check them."
                    + flight_context
            )
        else:
            prefixed_context = None

        result = run_reasoning_agent(
            request.message,
            request.history,
            map_image_base64=map_image_base64,
            flight_context=prefixed_context
        )

        return result

    except Exception as e:
        logger.error(f"[REASONING API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/flights/{flight_id}/ai-classification")
def get_flight_ai_classification(
    flight_id: str,
    schema: str = Query(default="live", description="Database schema to query")
):
    """
    Get AI classification for a specific flight.
    
    Args:
        flight_id: The flight identifier
        schema: Database schema (default: live)
    
    Returns:
        AI classification data including classification text, confidence, processing time, etc.
    """
    try:
        from service.pg_provider import get_connection
        
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(f"""
                    SELECT 
                        flight_id,
                        classification_text,
                        confidence_score,
                        processing_time_sec,
                        created_at,
                        error_message,
                        gemini_model
                    FROM {schema}.ai_classifications
                    WHERE flight_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (flight_id,))
                
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"AI classification not found for flight {flight_id}"
                    )
                
                return {
                    "flight_id": result["flight_id"],
                    "classification": result["classification_text"],
                    "confidence_score": result["confidence_score"],
                    "processing_time_sec": result["processing_time_sec"],
                    "created_at": result["created_at"].isoformat() if result["created_at"] else None,
                    "error": result["error_message"],
                    "model": result["gemini_model"]
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching AI classification for {flight_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ClassifyFlightByIdRequest(BaseModel):
    """Request for classifying a single flight by ID."""
    flight_id: str = Field(description="Flight identifier to classify")
    schema: str = Field(default="research", description="Database schema to use")
    force_reclassify: bool = Field(default=False, description="Force re-classification even if already classified")


class ClassifyFlightsByDateRangeRequest(BaseModel):
    """Request for classifying flights within a date range."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    schema: str = Field(default="research", description="Database schema to use")
    limit: int = Field(default=100, description="Maximum number of flights to classify")
    force_reclassify: bool = Field(default=False, description="Force re-classification even if already classified")


@router.post("/api/ai/classify-flight")
def classify_flight_by_id(request: ClassifyFlightByIdRequest):
    """
    Classify a single flight by flight ID.
    
    Retrieves flight data from the database, sends to AI for classification,
    and writes the result back to the database.
    
    Args:
        request: ClassifyFlightByIdRequest with flight_id, schema, and force_reclassify
    
    Returns:
        Classification result with flight_id, classification_text, processing time, etc.
    """
    try:
        import os
        from ai_classify import AIClassifier
        from service.pg_provider import (
            fetch_flight_data_for_classification,
            create_ai_classifications_table,
            get_connection
        )
        
        logger.info(f"Starting classification for flight {request.flight_id} in schema {request.schema}")
        
        # Check if already classified (unless force_reclassify is True)
        if not request.force_reclassify:
            try:
                with get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(f"""
                            SELECT COUNT(*) 
                            FROM {request.schema}.ai_classifications
                            WHERE flight_id = %s
                        """, (request.flight_id,))
                        count = cursor.fetchone()[0]
                        
                        if count > 0:
                            logger.info(f"Flight {request.flight_id} already classified, skipping")
                            return {
                                "success": True,
                                "skipped": True,
                                "message": f"Flight {request.flight_id} already classified. Use force_reclassify=true to override."
                            }
            except Exception as e:
                logger.warning(f"Error checking classification status: {e}")
        
        # Ensure table exists
        create_ai_classifications_table(request.schema)
        
        # Fetch flight data
        logger.info(f"Fetching data for flight {request.flight_id}...")
        flight_bundle = fetch_flight_data_for_classification(request.flight_id, request.schema)
        
        if not flight_bundle:
            raise HTTPException(
                status_code=404,
                detail=f"Flight {request.flight_id} not found or has no data"
            )
        
        # Initialize AI classifier
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBArSFAlxqm-9q1hWbaNgeT7f3WMOqF5Go")
        classifier = AIClassifier(api_key, schema=request.schema, max_workers=1)
        
        # Classify synchronously
        logger.info(f"Classifying flight {request.flight_id}...")
        start_time = time.time()
        
        result = classifier._classify_sync(
            flight_id=flight_bundle['flight_id'],
            flight_data=flight_bundle['flight_data'],
            anomaly_report=flight_bundle['anomaly_report'],
            metadata=flight_bundle['metadata']
        )
        
        # Shutdown classifier
        classifier.shutdown(wait=True)
        
        elapsed = time.time() - start_time
        
        if result.get('error_message'):
            logger.error(f"Classification failed: {result['error_message']}")
            return {
                "success": False,
                "flight_id": request.flight_id,
                "error": result['error_message'],
                "processing_time_sec": elapsed
            }
        
        logger.info(f"Classification completed: '{result.get('classification_text')}' ({elapsed:.2f}s)")
        
        return {
            "success": True,
            "flight_id": request.flight_id,
            "classification_text": result.get('classification_text'),
            "processing_time_sec": result.get('processing_time_sec'),
            "gemini_model": result.get('gemini_model'),
            "message": "Flight classified successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying flight {request.flight_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/classify-flights-by-date")
def classify_flights_by_date_range(request: ClassifyFlightsByDateRangeRequest):
    """
    Classify multiple flights within a date range.
    
    Retrieves unclassified anomaly flights from the specified date range,
    sends each to AI for classification, and writes results to the database.
    
    Args:
        request: ClassifyFlightsByDateRangeRequest with start_date, end_date, schema, limit, and force_reclassify
    
    Returns:
        Summary with total flights, success count, failed count, and processing time
    """
    try:
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ai_classify import AIClassifier
        from service.pg_provider import (
            fetch_flights_in_date_range,
            fetch_flight_data_for_classification,
            create_ai_classifications_table,
            get_connection
        )
        
        logger.info(f"Starting batch classification for date range {request.start_date} to {request.end_date}")
        
        # Ensure table exists
        create_ai_classifications_table(request.schema)
        
        # Fetch flight IDs in date range
        logger.info(f"Fetching unclassified flights in date range...")
        flight_ids = fetch_flights_in_date_range(
            request.start_date,
            request.end_date,
            request.schema,
            request.limit
        )
        
        if not flight_ids:
            return {
                "success": True,
                "total": 0,
                "classified": 0,
                "skipped": 0,
                "failed": 0,
                "message": "No unclassified anomalies found in the specified date range"
            }
        
        logger.info(f"Found {len(flight_ids)} flights to classify")
        
        # Initialize AI classifier
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBArSFAlxqm-9q1hWbaNgeT7f3WMOqF5Go")
        classifier = AIClassifier(api_key, schema=request.schema, max_workers=1)
        
        # Track results
        results = {
            'total': len(flight_ids),
            'classified': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        # Process flights with ThreadPoolExecutor
        max_workers = 2  # Parallel processing
        
        def classify_single_flight(flight_id: str) -> Dict[str, Any]:
            """Helper function to classify a single flight."""
            try:
                # Check if already classified
                if not request.force_reclassify:
                    with get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(f"""
                                SELECT COUNT(*) 
                                FROM {request.schema}.ai_classifications
                                WHERE flight_id = %s
                            """, (flight_id,))
                            count = cursor.fetchone()[0]
                            
                            if count > 0:
                                logger.info(f"Flight {flight_id} already classified, skipping")
                                return {'flight_id': flight_id, 'skipped': True, 'success': True}
                
                # Fetch flight data
                logger.info(f"Fetching data for flight {flight_id}...")
                flight_bundle = fetch_flight_data_for_classification(flight_id, request.schema)
                
                if not flight_bundle:
                    return {
                        'flight_id': flight_id,
                        'success': False,
                        'error': 'Flight data not found'
                    }
                
                # Classify
                logger.info(f"Classifying flight {flight_id}...")
                result = classifier._classify_sync(
                    flight_id=flight_bundle['flight_id'],
                    flight_data=flight_bundle['flight_data'],
                    anomaly_report=flight_bundle['anomaly_report'],
                    metadata=flight_bundle['metadata']
                )
                
                if result.get('error_message'):
                    return {
                        'flight_id': flight_id,
                        'success': False,
                        'error': result['error_message']
                    }
                
                logger.info(f"Successfully classified {flight_id}: '{result.get('classification_text')}'")
                return {
                    'flight_id': flight_id,
                    'success': True,
                    'classification_text': result.get('classification_text')
                }
                
            except Exception as e:
                logger.error(f"Error classifying flight {flight_id}: {e}")
                return {
                    'flight_id': flight_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Process flights in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_flight = {
                executor.submit(classify_single_flight, fid): fid
                for fid in flight_ids
            }
            
            for future in as_completed(future_to_flight):
                flight_id = future_to_flight[future]
                try:
                    result = future.result()
                    
                    if result.get('skipped'):
                        results['skipped'] += 1
                    elif result.get('success'):
                        results['classified'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append({
                            'flight_id': flight_id,
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    logger.error(f"Exception processing flight {flight_id}: {e}")
                    results['failed'] += 1
                    results['errors'].append({
                        'flight_id': flight_id,
                        'error': str(e)
                    })
        
        # Cleanup
        classifier.shutdown(wait=True)
        
        elapsed = time.time() - start_time
        
        logger.info(f"Batch classification complete: {results['classified']} classified, {results['skipped']} skipped, {results['failed']} failed in {elapsed:.1f}s")
        
        return {
            "success": True,
            "total": results['total'],
            "classified": results['classified'],
            "skipped": results['skipped'],
            "failed": results['failed'],
            "processing_time_sec": elapsed,
            "errors": results['errors'][:10] if results['errors'] else [],  # Return first 10 errors
            "message": f"Classified {results['classified']} flights, skipped {results['skipped']}, failed {results['failed']}"
        }
        
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
