-- Add formal structure columns to net_sessions table
-- Run this SQL script to manually add the columns needed for net detection overhaul

USE repeater;

-- Add formal_structure column (JSON)
ALTER TABLE net_sessions 
ADD COLUMN formal_structure JSON NULL AFTER source_segments;

-- Add ncs_script column (JSON array)
ALTER TABLE net_sessions 
ADD COLUMN ncs_script JSON NULL AFTER formal_structure;

-- Add checkin_sequence column (JSON array)
ALTER TABLE net_sessions 
ADD COLUMN checkin_sequence JSON NULL AFTER ncs_script;

-- Verify columns were added
DESCRIBE net_sessions;
