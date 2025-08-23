# Large Database & Cloud Storage Guide

## Handling 30GB+ iMessage Databases

When your iMessage database is very large (30GB+) and partially stored in iCloud, you need special handling to access all your data.

## 1. Understanding iMessage Cloud Storage

macOS stores iMessage data in two ways:
- **Local Storage**: Recent messages in `~/Library/Messages/chat.db`
- **iCloud Storage**: Older messages offloaded to iCloud to save space

### Check Your Current Status

```bash
# Check database size
ls -lh ~/Library/Messages/chat.db

# Check if Messages in iCloud is enabled
defaults read com.apple.MobileSMS | grep -i cloud
```

## 2. Download Full Database from iCloud

### Option A: Force Full Download (Recommended)
```bash
# 1. Open Messages app
open -a Messages

# 2. Scroll through conversations to trigger downloads
# Or use this command to trigger full sync:
brctl download ~/Library/Messages/

# 3. Monitor download progress
brctl status ~/Library/Messages/
```

### Option B: Disable Messages in iCloud Temporarily
1. Open System Preferences → Apple ID → iCloud
2. Uncheck "Messages" 
3. Wait for full database download (may take hours)
4. Re-enable after analysis if desired

## 3. Query Database for Available Data

Use our health check tool to see what's available:

```python
# Check database status
from mcp_server.tools.health import health_check_tool
import asyncio

async def check_db():
    result = await health_check_tool('~/Library/Messages/chat.db')
    print(f"Database size: {result['stats']['database_size_mb']} MB")
    print(f"Total messages: {result['stats']['total_messages']}")
    print(f"Date range: {result['stats']['date_range']}")
    
asyncio.run(check_db())
```

## 4. Handle Large Databases with Sharding

For databases over 10GB, use our sharding utility:

### Analyze First
```bash
python scripts/shard_large_database.py --analyze
```

### Create Time-Based Shards
```bash
# Shard into 6-month chunks
python scripts/shard_large_database.py \
    --source ~/Library/Messages/chat.db \
    --shards-dir ~/.imessage_insights/shards \
    --months 6

# For 30GB database, use smaller chunks
python scripts/shard_large_database.py \
    --source ~/Library/Messages/chat.db \
    --shards-dir ~/.imessage_insights/shards \
    --months 3 \
    --force
```

### Run Server with Shards
```bash
# Start server with sharding support
python server.py \
    --use-shards \
    --shards-dir ~/.imessage_insights/shards \
    --disable-memory-monitor
```

## 5. Optimize for Large Databases

### Create Indexes
```bash
python scripts/index_imessage_db.py
```

### Use Streaming Queries
When using tools, limit time ranges:
```python
# Instead of analyzing all data at once
result = await imsg_summary_overview(db_path="~/Library/Messages/chat.db")

# Process in chunks
for year in range(2020, 2025):
    result = await imsg_relationship_intelligence(
        contact_id="...",
        window_days=365,
        start_date=f"{year}-01-01"
    )
```

## 6. Memory Management

### Monitor Memory Usage
```bash
# Check current memory usage
ps aux | grep python | grep server

# Run with memory limits
ulimit -v 8000000  # 8GB limit
python server.py
```

### Configure Batch Sizes
```json
{
  "performance": {
    "batch_size": 1000,
    "cache_size": 100,
    "streaming_threshold": 10000
  }
}
```

## 7. Cloud-Aware Querying

### Check Message Availability
```sql
-- Find messages that might be in cloud
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) as cloud_messages,
    SUM(CASE WHEN text IS NOT NULL THEN 1 ELSE 0 END) as local_messages
FROM message;

-- Find date ranges with missing data
SELECT 
    strftime('%Y-%m', datetime(date/1000000000 + 978307200, 'unixepoch')) as month,
    COUNT(*) as messages,
    SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) as missing
FROM message
GROUP BY month
ORDER BY month DESC;
```

### Progressive Loading
```python
async def analyze_with_cloud_awareness(db_path, contact_id):
    # First, check what's available locally
    local_stats = await check_local_availability(db_path, contact_id)
    
    if local_stats['missing_percentage'] > 20:
        print(f"Warning: {local_stats['missing_percentage']}% of messages are in iCloud")
        print("Consider downloading full database first")
    
    # Analyze available data
    return await analyze_available_messages(db_path, contact_id)
```

## 8. Best Practices

### For 30GB+ Databases:
1. **Always shard first** - Don't try to load entire database into memory
2. **Use time-based analysis** - Process year by year or month by month
3. **Enable streaming** - Use streaming queries for large result sets
4. **Monitor resources** - Watch CPU, memory, and disk I/O
5. **Backup first** - Always backup before sharding or indexing

### Performance Tips:
- Close other apps to free memory
- Use SSD storage for shards directory
- Run analysis during off-hours
- Consider using a dedicated machine

## 9. Troubleshooting

### Common Issues:

**"Database locked" errors**
```bash
# Kill any processes using the database
lsof | grep chat.db
kill -9 <PID>
```

**Out of memory errors**
```bash
# Increase swap space
sudo sysctl vm.swapusage
```

**Slow queries**
```bash
# Vacuum and analyze database
sqlite3 ~/Library/Messages/chat.db "VACUUM; ANALYZE;"
```

## 10. Example: Full Analysis Workflow

```bash
# 1. Check current status
python scripts/shard_large_database.py --analyze

# 2. Download from cloud (if needed)
brctl download ~/Library/Messages/

# 3. Create shards
python scripts/shard_large_database.py --months 3

# 4. Index for performance  
python scripts/index_imessage_db.py

# 5. Run server with shards
python server.py --use-shards --shards-dir ~/.imessage_insights/shards

# 6. Use tools with time ranges
# In Claude:
# "Analyze my conversations with John from 2023"
# "Show communication patterns for Q1 2024"
```

## Additional Resources

- [Apple Support: Messages in iCloud](https://support.apple.com/en-us/HT208532)
- [SQLite Performance Tuning](https://www.sqlite.org/optoverview.html)
- [Memory Management Best Practices](https://docs.python.org/3/library/resource.html)