# Node Selection for Manual Net Control - Implementation

## Overview
Added ability to specify which node to monitor when manually starting a net through the web portal.

## Changes Made

### 1. Frontend Form (`web/templates/net_control.html`)
**Added node selector dropdown** to the "Flag Net Start" form:
```html
<label for="net-node">Node (optional)</label>
<select id="net-node" name="node_id">
    <option value="">All Nodes</option>
    {% for node in available_nodes %}
        <option value="{{ node }}" {% if selected_node == node %}selected{% endif %}>{{ node }}</option>
    {% endfor %}
</select>
```

**Features**:
- Only shows if multiple nodes are available
- Pre-selects the currently filtered node (if viewing a specific node)
- Optional - defaults to "All Nodes"

### 2. Backend Route (`web/routes/profiles.py`)
**Added node_id parameter** to form handler:
```python
@router.post("/net-control/start")
async def start_net_control(
    name: str = Form(""),
    notes: str = Form(""),
    node_id: str = Form(""),  # NEW
    ...
):
    payload = await net_control.start_session(
        ...
        node_id=node_id or None,  # Pass to service
    )
```

### 3. Service Layer (`web/services/net_control.py`)
**Updated start_session()** to accept and store node_id:
```python
async def start_session(
    session,
    *,
    name: str,
    started_by: str,
    started_by_callsign: str | None,
    notes: str | None = None,
    node_id: str | None = None,  # NEW parameter
) -> dict:
    # Store node_id in metadata
    metadata = {}
    if node_id:
        metadata["node_id"] = node_id
    
    record = NetControlSession(
        ...
        metadata_=metadata,
    )
```

**Updated serialize_session()** to return node_id:
```python
def serialize_session(record: NetControlSession | None) -> dict | None:
    # Extract node_id from metadata
    node_id = None
    if record.metadata_:
        node_id = record.metadata_.get("node_id")
    
    return {
        ...
        "node_id": node_id,  # NEW field
    }
```

### 4. Display Updates (`web/templates/net_control.html`)
**Active session shows node**:
```html
<p class="muted">
    Active since {{ active_session.started_at }} • started by {{ active_session.started_by }}
    {% if active_session.node_id %}
        • <span class="node-badge">Node: {{ active_session.node_id }}</span>
    {% endif %}
</p>
```

**Recent sessions show node**:
```html
<span>
    {{ session.name }} • {{ session.started_at }}
    {% if session.node_id %}
        • <span class="node-badge">{{ session.node_id }}</span>
    {% endif %}
</span>
```

## User Experience

### Before
```
Manual Net Session
┌─────────────────────────────┐
│ Net name: [              ] │
│ Notes:    [              ] │
│           [Flag Net Start] │
└─────────────────────────────┘
```

### After
```
Manual Net Session
┌─────────────────────────────┐
│ Net name: [              ] │
│ Node:     [All Nodes ▼   ] │  ← NEW
│           [node_1        ] │
│           [node_2        ] │
│ Notes:    [              ] │
│           [Flag Net Start] │
└─────────────────────────────┘
```

### Active Session Display
**Before**: `Active since 2026-01-28 10:00 • started by W7XYZ`
**After**: `Active since 2026-01-28 10:00 • started by W7XYZ • Node: node_1`

## Data Storage

Node selection is stored in `net_control_sessions.metadata` JSON field:
```json
{
  "node_id": "node_1"
}
```

This approach:
- ✅ No database migration required (uses existing metadata column)
- ✅ Backward compatible (null/empty metadata = all nodes)
- ✅ Extensible (can add more metadata later)

## Use Cases

1. **Single Node Net**: Operator starts net and selects `node_1` → only node_1 transcripts monitored
2. **Multi-Node Net**: Operator starts net with "All Nodes" → all nodes monitored
3. **Node Switching**: Operator can view different nodes using the top filter while net is active

## Integration with Existing Features

- **Node Filter** at top of page: Filters display of live callsigns and transcripts
- **Manual Net Start**: Now respects node selection for net scope
- **Live Callsigns**: Already shows node badges, now filtered by selected node
- **Recent Sessions**: Shows which node each manual net was scoped to

## Testing

1. **Start net with node selection**:
   - Navigate to `/net-control`
   - Fill net name
   - Select specific node from dropdown
   - Click "Flag Net Start"
   - Verify active session shows node badge

2. **Start net without node (all nodes)**:
   - Leave node dropdown on "All Nodes"
   - Start net
   - Verify no node badge shown (or shows "All Nodes")

3. **View recent sessions**:
   - Check recent sessions list
   - Verify node badges appear for node-specific nets
   - Verify no badge for all-node nets

## Files Modified
1. `web/templates/net_control.html` - Added form field + display badges
2. `web/routes/profiles.py` - Added node_id form parameter
3. `web/services/net_control.py` - Added node_id handling in start_session() and serialize_session()

**Total**: 3 files, backward compatible, no database changes needed
