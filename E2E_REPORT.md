# E2E API Validation Report

## API Server is UP

### 1. Upload Video
**PASS**: Uploaded Video ID: `9049dd31-d522-4f91-a7ff-fc1f88ff842c`

### 2. Extract Frames (Native HF)
**PASS**: Extracted frames in 215.29s
Candidate Frames: 5
Metadata Success: 0
Metadata Failed: 5

### 3. Generate Summaries (Event Aggregation)
**FAIL**: Summaries failed 405 {"detail":"Method Not Allowed"}

### 4. Generate Incidents
**FAIL**: Incidents failed 405 {"detail":"Method Not Allowed"}

### 5. Generate Poster
**FAIL**: Poster failed 405 {"detail":"Method Not Allowed"}

### 6. Verify Search Indexing
**FAIL**: Search verification failed: No module named 'app.core.qdrant_manager'
