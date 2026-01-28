# ✅ Learned Data Upload - Implementation Checklist

Use this checklist to implement the PostgreSQL learned data upload system.

## 📋 Pre-Implementation

- [ ] PostgreSQL is installed and running
- [ ] `POSTGRES_DSN` is set in `.env` file
- [ ] `psycopg2-binary` is in `requirements.txt` (already ✅)
- [ ] JSON files exist in `rules/` directory:
  - [ ] `learned_paths.json`
  - [ ] `learned_tubes.json`
  - [ ] `learned_sid.json`
  - [ ] `learned_star.json`

## 🚀 Initial Upload

- [ ] Test PostgreSQL connection:
  ```bash
  python -c "from service.pg_provider import test_connection; test_connection()"
  ```

- [ ] Run the upload script:
  ```bash
  python scripts/upload_learned_data_to_postgres.py
  ```

- [ ] Verify upload succeeded (check output for "✅ SUCCESS!")

- [ ] Run test script:
  ```bash
  python scripts/test_upload.py
  ```

- [ ] Verify all tests passed (6/6)

## 🔍 Verification

- [ ] Check database statistics:
  ```python
  from service.learned_data_provider import get_statistics
  stats = get_statistics()
  print(f"Paths: {stats['paths']['total']}")
  print(f"Tubes: {stats['tubes']['total']}")
  print(f"SIDs: {stats['sids']['total']}")
  print(f"STARs: {stats['stars']['total']}")
  ```

- [ ] Test a sample query:
  ```python
  from service.learned_data_provider import get_paths_by_route
  paths = get_paths_by_route("LLBG", "LLSD")
  print(f"Found {len(paths)} paths")
  ```

- [ ] Verify query performance (should be < 10ms)

## 🔄 Code Migration (if replacing MongoDB)

- [ ] Read the migration guide (`MIGRATION_GUIDE.md`)

- [ ] Identify all MongoDB queries in your code:
  - [ ] Search for `from core.mongo_queries import`
  - [ ] Search for `find_paths_by_route`
  - [ ] Search for `find_tubes_by_route`
  - [ ] Search for `db["learned_tubes"].find`

- [ ] Update imports to use PostgreSQL provider:
  ```python
  # Old
  from core.mongo_queries import find_paths_by_route
  
  # New
  from service.learned_data_provider import get_paths_by_route
  ```

- [ ] Update query calls (see MIGRATION_GUIDE.md for examples)

- [ ] Test each updated endpoint

- [ ] Compare results with MongoDB (optional verification)

## 📝 API Integration

- [ ] Update `/api/learned-layers` endpoint to use PostgreSQL:
  ```python
  from service.learned_data_provider import get_all_learned_layers
  ```

- [ ] Update `/api/paths` endpoint (if exists)

- [ ] Update `/api/union-tubes` endpoint (if exists)

- [ ] Test all endpoints:
  ```bash
  curl "http://localhost:8000/api/learned-layers?origin=LLBG&destination=LLSD"
  ```

- [ ] Verify response times improved

- [ ] Check API logs for errors

## 🧪 Testing

- [ ] Unit tests pass (if applicable)

- [ ] Integration tests pass (if applicable)

- [ ] Manual testing:
  - [ ] Query by origin only
  - [ ] Query by destination only
  - [ ] Query by origin-destination pair
  - [ ] Query with high min_member_count
  - [ ] Query with low min_member_count
  - [ ] Get all data (no filters)

- [ ] Load testing (optional):
  ```bash
  # Test response time
  time curl "http://localhost:8000/api/learned-layers" -s > /dev/null
  ```

## 📊 Monitoring

- [ ] Add query performance logging:
  ```python
  import time
  start = time.time()
  paths = get_paths_by_route("LLBG", "LLSD")
  logger.info(f"Query took {(time.time() - start) * 1000:.2f}ms")
  ```

- [ ] Monitor PostgreSQL connection pool:
  ```python
  from service.pg_provider import get_pool
  pool = get_pool()
  # Check pool status in health endpoint
  ```

- [ ] Set up alerts for slow queries (> 100ms)

## 🔧 Optimization (Optional)

- [ ] Run ANALYZE on tables:
  ```sql
  ANALYZE learned_paths;
  ANALYZE learned_tubes;
  ANALYZE learned_sids;
  ANALYZE learned_stars;
  ```

- [ ] Check query plans:
  ```sql
  EXPLAIN ANALYZE 
  SELECT * FROM learned_paths 
  WHERE origin = 'LLBG' AND destination = 'LLSD';
  ```

- [ ] Add custom indexes if needed (see README)

## 📚 Documentation

- [ ] Update API documentation with new endpoints

- [ ] Document database schema (already in README)

- [ ] Update deployment documentation

- [ ] Add to ops runbook:
  - How to upload data
  - How to verify uploads
  - How to troubleshoot

## 🚢 Deployment

- [ ] Update environment variables in production:
  ```
  POSTGRES_DSN=postgresql://user:pass@prod-host:5432/db
  ```

- [ ] Run upload script in production:
  ```bash
  python scripts/upload_learned_data_to_postgres.py
  ```

- [ ] Verify production upload

- [ ] Deploy updated code

- [ ] Monitor logs for errors

- [ ] Check performance metrics

- [ ] Verify API responses

## 🎯 Post-Deployment

- [ ] Monitor query performance for 24-48 hours

- [ ] Check error rates

- [ ] Verify data consistency

- [ ] Update monitoring dashboards

- [ ] Document any issues encountered

- [ ] Create backup/restore procedure:
  ```bash
  # Backup
  pg_dump -t learned_paths -t learned_tubes -t learned_sids -t learned_stars > learned_data_backup.sql
  
  # Restore
  psql < learned_data_backup.sql
  ```

## 🗑️ Cleanup (After Stable Period)

- [ ] Remove MongoDB queries from codebase

- [ ] Remove MongoDB dependencies

- [ ] Update requirements.txt

- [ ] Remove `core/mongo_queries.py` (if no longer needed)

- [ ] Update documentation to remove MongoDB references

## 🎉 Complete!

Once all checkboxes are checked, you've successfully migrated to PostgreSQL for learned data!

## 📞 Need Help?

- **Quick start**: `QUICKSTART.md`
- **Full docs**: `README_LEARNED_DATA_UPLOAD.md`
- **Migration**: `MIGRATION_GUIDE.md`
- **Reference**: `QUICK_REFERENCE.md`
- **Overview**: `SUMMARY.md`

---

**Tip**: You don't have to do everything at once. Start with the upload, verify it works, then gradually migrate your code.
