# Marine API Tests

Comprehensive test suite for the marine vessel tracking API endpoints.

## Test Coverage

The test suite covers all 5 marine API endpoints:

1. **GET /api/marine/vessels** - List active vessels
   - Default parameters
   - Pagination (limit/offset)
   - Time filtering (since)
   - Type filtering
   - Bounding box filtering
   - Response structure validation

2. **GET /api/marine/vessels/{mmsi}** - Get vessel details
   - Valid MMSI lookup
   - 404 handling for non-existent vessels
   - Complete response structure validation

3. **GET /api/marine/vessels/{mmsi}/track** - Get vessel track history
   - Track retrieval with default parameters
   - Custom time window (hours parameter)
   - 404 handling
   - Track point structure validation

4. **GET /api/marine/search** - Search vessels
   - Search by vessel name
   - Search by MMSI
   - Minimum length validation
   - Missing query parameter handling

5. **GET /api/marine/statistics** - Get traffic statistics
   - Default statistics (1 hour)
   - Custom time period
   - Response structure validation

## Running the Tests

### Prerequisites

1. Ensure PostgreSQL is running and accessible
2. Set `POSTGRES_DSN` environment variable in `.env` file
3. Ensure marine data exists in the database (run `marine_monitor.py` first)
4. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio
   ```

### Run All Marine Tests

```bash
cd tracer-api
python -m pytest tests/test_marine.py -v
```

### Run Specific Test

```bash
python -m pytest tests/test_marine.py::test_get_vessels_default_params -v
```

### Run with Coverage

```bash
python -m pytest tests/test_marine.py --cov=service.marine_service --cov=routes.marine_routes -v
```

### Run Tests in Parallel

```bash
pip install pytest-xdist
python -m pytest tests/test_marine.py -n auto -v
```

## Test Output Example

```
tests/test_marine.py::test_get_vessels_default_params PASSED
tests/test_marine.py::test_get_vessels_with_limit_offset PASSED
tests/test_marine.py::test_get_vessels_with_since_filter PASSED
tests/test_marine.py::test_get_vessel_by_mmsi PASSED
tests/test_marine.py::test_get_vessel_track PASSED
tests/test_marine.py::test_search_vessels PASSED
tests/test_marine.py::test_get_statistics PASSED
```

## Test Data Requirements

Some tests will be skipped if:
- No vessels exist in the database
- No vessels with names exist
- No recent position data exists

To ensure all tests run:
1. Start the marine monitor: `python tracer-worker/run_marine_monitor.py`
2. Wait a few minutes for data to be collected
3. Run the tests

## Troubleshooting

### Database Connection Errors

If you see connection errors:
```bash
# Check your .env file has correct POSTGRES_DSN
cat .env | grep POSTGRES_DSN

# Test connection manually
python -c "from service.pg_provider import get_connection; print('Connected:', get_connection().__enter__())"
```

### No Data Found

If tests skip due to no data:
```bash
# Check if marine data exists
psql $POSTGRES_DSN -c "SELECT COUNT(*) FROM marine.vessel_positions;"

# Start the marine monitor
cd tracer-worker
python run_marine_monitor.py
```

### Import Errors

If you see import errors:
```bash
# Make sure you're in the tracer-api directory
cd tracer-api

# Run tests with proper module resolution
python -m pytest tests/test_marine.py
```

## Manual API Testing

You can also test the API manually using curl:

```bash
# List active vessels
curl http://localhost:8000/api/marine/vessels?limit=10

# Get vessel details
curl http://localhost:8000/api/marine/vessels/368207620

# Get vessel track
curl http://localhost:8000/api/marine/vessels/368207620/track?hours=24

# Search vessels
curl http://localhost:8000/api/marine/search?q=EVER

# Get statistics
curl http://localhost:8000/api/marine/statistics?since=1
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Test Marine API
  run: |
    cd tracer-api
    python -m pytest tests/test_marine.py -v --junitxml=marine-test-results.xml
```

## Test Maintenance

When adding new endpoints or modifying existing ones:

1. Add corresponding tests to `test_marine.py`
2. Update this README with new test cases
3. Ensure all tests pass before committing
4. Update coverage requirements if needed
