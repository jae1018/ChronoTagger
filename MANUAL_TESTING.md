# Phase 2 Manual Testing Checklist

## Setup
Test data files are in `test_data/`:
- `test_data.csv` - CSV with DatetimeIndex
- `test_data.parquet` - Parquet file
- `test_data_no_index.csv` - CSV without DatetimeIndex (requires auto-detection)
- `test_invalid.csv` - Invalid data (no numeric columns)

## Launch Command
```bash
chronotagger
```

## Test Cases

### 1. Basic CSV Loading
- [ ] Click "Browse..." button
- [ ] Select `test_data/test_data_no_index.csv`
- [ ] Verify preview shows first 10 rows
- [ ] Verify "Time Column" dropdown shows "timestamp" selected
- [ ] Verify status shows "Rows: 120, Columns: 4"
- [ ] Verify "Continue" button is enabled
- [ ] Click "Continue"
- [ ] Verify success message shows data loaded with correct dimensions

### 2. Parquet File Loading
- [ ] Launch `chronotagger`
- [ ] Click "Browse..."
- [ ] Select `test_data/test_data.parquet`
- [ ] Verify data preview appears
- [ ] Verify time column detected
- [ ] Click "Continue"
- [ ] Verify success

### 3. Auto-detect Time Column
- [ ] Launch `chronotagger`
- [ ] Load `test_data_no_index.csv`
- [ ] Verify "Time Column" dropdown automatically selects "timestamp"
- [ ] Verify preview updates correctly

### 4. Manual Time Column Selection
- [ ] Launch `chronotagger`
- [ ] Load `test_data_no_index.csv`
- [ ] Change "Time Column" dropdown to different column (e.g., "BX")
- [ ] Verify "Continue" button becomes disabled (invalid column)
- [ ] Change back to "timestamp"
- [ ] Verify "Continue" button becomes enabled

### 5. Invalid File Handling
- [ ] Launch `chronotagger`
- [ ] Load `test_data/test_invalid.csv`
- [ ] Verify error appears or "Continue" is disabled
- [ ] Verify status shows error message: "No numeric columns found"

### 6. Cancel Dialog
- [ ] Launch `chronotagger`
- [ ] Click "Cancel" button without loading file
- [ ] Verify confirmation dialog appears
- [ ] Click "Yes"
- [ ] Verify application exits

### 7. Cancel After Loading
- [ ] Launch `chronotagger`
- [ ] Load valid file
- [ ] Click "Cancel"
- [ ] Verify confirmation dialog appears
- [ ] Click "Yes"
- [ ] Verify application exits

### 8. Browse and Cancel File Dialog
- [ ] Launch `chronotagger`
- [ ] Click "Browse..."
- [ ] Cancel the file selection dialog
- [ ] Verify application remains open and functional

### 9. Data Preview Scrolling
- [ ] Launch `chronotagger`
- [ ] Load `test_data_no_index.csv`
- [ ] Verify horizontal scrollbar appears (multiple columns)
- [ ] Scroll horizontally to view all columns
- [ ] Verify all 5 columns visible: timestamp, log10n, BX, BY, BZ

### 10. File Path Display
- [ ] Launch `chronotagger`
- [ ] Load any file
- [ ] Verify full file path displayed in "File:" field
- [ ] Verify path is read-only (cannot edit directly)

## Success Criteria
All checkboxes above should be checked for Phase 2 to be considered complete.

## Known Limitations (Expected)
- Phase 3 (column selection) not implemented - shows stub message
- Only CSV and Parquet files supported
- No HDF5 support (deferred to future phase)
