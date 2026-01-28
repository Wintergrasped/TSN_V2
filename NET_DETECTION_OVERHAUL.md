# Net Detection System Overhaul - Complete

## Problem Statement
The original net detection system was too generic and didn't leverage the highly structured format of amateur radio nets, which ALWAYS have:
- **Formal Opening**: NCS announces net name, club, and purpose
- **Formal Check-ins**: Roll call with callsigns, locations, and names
- **Formal Closing**: Sign-off procedures and closing statements

## Changes Made

### 1. Enhanced vLLM Detection Prompt (`tsn_server/analyzer.py`)
**OLD PROMPT** (lines ~1188-1197):
```
Tasks (minimum requirements):
1. Net Identification: Decide whether the provided transcripts form one cohesive net
2. Net Validation: Justify why the evidence proves it is a legitimate session
3-5. Other analysis tasks...
```

**NEW PROMPT**:
```
CRITICAL: Amateur radio NETS have FORMAL STRUCTURE - detect these specific markers:

1. NET DETECTION (Primary Goal):
   FORMAL NETS ALWAYS HAVE:
   a) OPENING: NCS announces net name, club, purpose ("This is the [name] net")
   b) CHECK-INS: Formal roll call with callsigns, locations, names
      - NCS says: "Any check-ins?", "Go ahead with your call", "We have [callsign]"
      - Stations say: "This is [callsign], [name], [location]"
   c) CLOSING: Formal sign-off ("This concludes the [name] net", "73s", etc.)
   
   Extract these MANDATORY fields for each net:
   - "formal_structure": {opening/checkins/closing booleans + exact quotes}
   - "ncs_script": [List of NCS management statements]
   - "checkin_sequence": [Ordered list of formal check-ins with quotes]
```

**Impact**: AI now explicitly looks for formal net structures instead of guessing.

### 2. Enhanced JSON Schema (`tsn_server/analyzer.py`)
**NEW FIELDS ADDED**:
```json
{
  "nets": [{
    "formal_structure": {
      "has_opening": true|false,
      "has_checkins": true|false,
      "has_closing": true|false,
      "opening_text": "Exact quote from transcript",
      "closing_text": "Exact quote from transcript",
      "opening_source": transcript_number,
      "closing_source": transcript_number
    },
    "ncs_script": ["List of NCS statements like 'This is...', 'Any check-ins?', etc"],
    "checkin_sequence": [
      {"callsign": "W1ABC", "time": transcript_number, "statement": "exact check-in quote"}
    ],
    "stats": {
      "formal_checkins": 0  // NEW: Count of formal check-ins
    },
    "participants": [
      {"checkin_statement": "exact quote"}  // NEW: Quote for each participant
    ]
  }]
}
```

### 3. Database Model Updates (`tsn_common/models/net.py`)
**NEW COLUMNS ADDED TO `net_sessions` TABLE**:
```python
# Formal structure detection (JSON fields)
formal_structure: Mapped[dict | None] = mapped_column(JSON, nullable=True)
ncs_script: Mapped[list | None] = mapped_column(JSON, nullable=True)
checkin_sequence: Mapped[list | None] = mapped_column(JSON, nullable=True)
```

**NOTE**: SQLAlchemy auto-migration will add these columns on next startup. No manual migration needed.

### 4. Enhanced Validation Pass (`tsn_server/analyzer.py`)
**OLD VALIDATION**:
```python
"Given transcript digests and proposed nets, mark each net as valid=true|false"
```

**NEW VALIDATION**:
```python
"FORMAL NETS must have: opening statement, formal check-ins, and closing. "
"Validate each net checking: (1) Has complete formal structure? "
"(2) Contains formal check-in statements? (3) Confidence >=0.85? "
"Mark as valid=true|false with confidence 0-1 and specific reason. "
"Reject if structure is incomplete or check-ins are vague."
```

**NEW PRE-VALIDATION LOGIC**:
```python
# Add formal structure info to validation
for i, net in enumerate(nets):
    if i < len(nets_outline):
        formal = net.get("formal_structure", {})
        nets_outline[i]["has_formal_structure"] = all([
            formal.get("has_opening"),
            formal.get("has_checkins"),
            formal.get("has_closing")
        ])
        nets_outline[i]["formal_checkins"] = len(net.get("checkin_sequence", []))
```

**Impact**: Validator now checks for formal structure, not just generic confidence.

### 5. Persistence Layer Updates (`tsn_server/analyzer.py`)
**BEFORE** (line ~1851):
```python
net_session = NetSession(
    net_name=net.get("name", "Unnamed Net"),
    # ... standard fields ...
    source_segments=source_segments,
)
```

**AFTER**:
```python
# Extract formal structure data
formal_structure = net.get("formal_structure")
ncs_script = net.get("ncs_script")
checkin_sequence = net.get("checkin_sequence")

net_session = NetSession(
    net_name=net.get("name", "Unnamed Net"),
    # ... standard fields ...
    source_segments=source_segments,
    formal_structure=formal_structure,
    ncs_script=ncs_script,
    checkin_sequence=checkin_sequence,
)
```

### 6. Web Display Enhancements

#### A. Net List Page (`web/templates/nets.html`)
**NEW COLUMN**: "Structure" with visual indicators:
```html
<td>
    {% if net.formal_structure %}
        {% set fs = net.formal_structure %}
        <span title="Opening: Yes/No, Check-ins: Yes/No, Closing: Yes/No">
            {{ '✓' if fs.get('has_opening') else '✗' }}
            {{ '✓' if fs.get('has_checkins') else '✗' }}
            {{ '✓' if fs.get('has_closing') else '✗' }}
        </span>
    {% endif %}
</td>
```

**NEW COLUMN**: "Formal Check-ins" showing count:
```html
<td>
    {% if net.checkin_sequence %}
        {{ net.checkin_sequence | length }}
    {% endif %}
</td>
```

**Before/After**:
```
BEFORE: Name | Club | Start | Duration | Participants | Avg Check-in | Confidence
AFTER:  Name | Club | Start | Duration | Structure | Participants | Formal Check-ins | Confidence
```

#### B. Net Detail Page (`web/templates/net_summary.html`)
**NEW SECTION 1**: "Formal Structure" box showing:
- ✓/✗ indicators for opening, check-ins, closing
- Opening statement quote with source transcript
- Closing statement quote with source transcript

**NEW SECTION 2**: "NCS Script" showing:
- First 10 NCS management statements
- Full list of how Net Control ran the session

**NEW SECTION 3**: "Formal Check-in Sequence" table:
```html
<table>
    <thead>
        <tr>
            <th>#</th>
            <th>Callsign</th>
            <th>Time</th>
            <th>Statement</th>
        </tr>
    </thead>
    <tbody>
        {% for checkin in net.checkin_sequence %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ callsign_link(checkin.callsign) }}</td>
            <td>Transcript #{{ checkin.time }}</td>
            <td><em>"{{ checkin.statement }}"</em></td>
        </tr>
        {% endfor %}
    </tbody>
</table>
```

### 7. Web Service Updates
**Updated Files**:
- `web/services/nets.py` - Added formal_structure, ncs_script, checkin_sequence to fetch_net_summary()
- `web/services/dashboard.py` - Added formal_structure, checkin_sequence to get_recent_nets()

## Expected Results

### Before Overhaul
- Generic detection: "These transcripts seem like a net"
- Display shows: Name, duration, participant count, vague confidence
- No visibility into formal net lifecycle
- False positives from random QSOs

### After Overhaul
- Specific detection: "Opening detected: '[Quote]', 8 formal check-ins, Closing: '[Quote]'"
- Display shows: Full structure visualization, NCS script, ordered check-in sequence
- Complete visibility: See exactly what makes it a net
- Higher confidence: Validation requires formal structure

## Testing Recommendations

1. **Run analyzer on existing transcripts** - Watch for new formal_structure data in logs
2. **Check database** - Columns will auto-add on next startup
3. **View nets page** - Should see ✓✗✓ structure indicators
4. **Click into net detail** - Should see opening/closing quotes and formal check-in table
5. **Monitor validation** - Should reject incomplete structures

## Rollback Plan
If issues occur:
1. Revert `tsn_server/analyzer.py` prompt changes
2. Database columns are nullable - old data still works
3. Templates gracefully handle missing data with `{% if %}` checks

## Migration Notes
- **Database**: No manual migration needed - SQLAlchemy auto-migration handles new columns
- **Existing Nets**: Will show "—" for formal structure (graceful degradation)
- **New Nets**: Will populate all new fields automatically

## Files Modified
1. `tsn_server/analyzer.py` - Prompt, schema, validation, persistence (4 changes)
2. `tsn_common/models/net.py` - Database model + to_dict() (2 changes)
3. `web/templates/nets.html` - List view with structure column (1 change)
4. `web/templates/net_summary.html` - Detail view with 3 new sections (1 change)
5. `web/services/nets.py` - fetch_net_summary() return data (1 change)
6. `web/services/dashboard.py` - get_recent_nets() return data (1 change)

**Total**: 10 changes across 6 files

## Success Metrics
- ✅ Nets now have explicit formal structure detection
- ✅ Display shows opening/closing quotes
- ✅ Formal check-in sequence visible with timestamps
- ✅ NCS script captured and displayed
- ✅ Validation enforces structural requirements
- ✅ Backward compatible - old nets still display

---

**Status**: ✅ COMPLETE - Ready for deployment and testing
