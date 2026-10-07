import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"

load_dotenv(BASE_DIR / ".env")


async def check_tower_status(tower_id: str) -> dict:
    """
    Get the current status of a network tower.

    Reads tower inventory, the latest performance sample,
    and any currently open incidents for the tower.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                t.tower_id,
                t.tower_name,
                t.region,
                t.city,
                t.state,
                t.technology,
                t.status,

                p.recorded_at,
                p.latency_ms,
                p.packet_loss_pct,
                p.downlink_throughput_mbps,
                p.uplink_throughput_mbps,
                p.signal_strength_dbm,
                p.active_connections,

                i.incident_id,
                i.severity AS incident_severity,
                i.status AS incident_status,
                i.title AS incident_title,
                i.description AS incident_description,
                i.opened_at AS incident_opened_at,
                i.classification AS incident_classification,
                i.assigned_team

            FROM network_towers AS t

            LEFT JOIN tower_performance AS p
                ON p.tower_id = t.tower_id
                AND p.recorded_at = (
                    SELECT MAX(p2.recorded_at)
                    FROM tower_performance AS p2
                    WHERE p2.tower_id = t.tower_id
                )

            LEFT JOIN open_incidents AS i
                ON i.tower_id = t.tower_id
                AND i.status != 'RESOLVED'

            WHERE t.tower_id = ?
            """,
            (tower_id,),
        )

        rows = cursor.fetchall()

        if not rows:
            return {
                "success": False,
                "tower_id": tower_id,
                "error": "Tower not found.",
            }

        first = rows[0]

        incidents = []

        for row in rows:
            if row["incident_id"] is not None:
                incidents.append(
                    {
                        "incident_id": row["incident_id"],
                        "severity": row["incident_severity"],
                        "status": row["incident_status"],
                        "title": row["incident_title"],
                        "description": row["incident_description"],
                        "opened_at": row["incident_opened_at"],
                        "classification": row[
                            "incident_classification"
                        ],
                        "assigned_team": row["assigned_team"],
                    }
                )

        return {
            "success": True,
            "tower": {
                "tower_id": first["tower_id"],
                "tower_name": first["tower_name"],
                "region": first["region"],
                "city": first["city"],
                "state": first["state"],
                "technology": first["technology"],
                "status": first["status"],
            },
            "latest_performance": {
                "recorded_at": first["recorded_at"],
                "latency_ms": first["latency_ms"],
                "packet_loss_pct": first["packet_loss_pct"],
                "downlink_throughput_mbps":
                    first["downlink_throughput_mbps"],
                "uplink_throughput_mbps":
                    first["uplink_throughput_mbps"],
                "signal_strength_dbm":
                    first["signal_strength_dbm"],
                "active_connections":
                    first["active_connections"],
            },
            "open_incidents": incidents,
        }

    finally:
        connection.close()


async def run_connectivity_diagnostics(
    tower_id: str,
    symptom: str,
) -> dict:
    """
    Run connectivity diagnostics for a tower.

    Use this when users report dropped sessions, slow speeds,
    packet loss, latency, weak signal, or other connectivity
    symptoms near a specific tower.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                t.tower_id,
                t.tower_name,
                t.region,
                t.city,
                t.state,
                t.technology,
                t.status,

                p.recorded_at,
                p.latency_ms,
                p.packet_loss_pct,
                p.downlink_throughput_mbps,
                p.uplink_throughput_mbps,
                p.signal_strength_dbm,
                p.active_connections

            FROM network_towers AS t

            LEFT JOIN tower_performance AS p
                ON p.tower_id = t.tower_id
                AND p.recorded_at = (
                    SELECT MAX(p2.recorded_at)
                    FROM tower_performance AS p2
                    WHERE p2.tower_id = t.tower_id
                )

            WHERE t.tower_id = ?
            """,
            (tower_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return {
                "success": False,
                "tower_id": tower_id,
                "error": "Tower not found.",
            }

        return {
            "success": True,
            "tower_id": row["tower_id"],
            "tower_name": row["tower_name"],
            "location": {
                "region": row["region"],
                "city": row["city"],
                "state": row["state"],
            },
            "technology": row["technology"],
            "tower_status": row["status"],
            "reported_symptom": symptom,
            "latest_performance": {
                "recorded_at": row["recorded_at"],
                "latency_ms": row["latency_ms"],
                "packet_loss_pct": row["packet_loss_pct"],
                "downlink_throughput_mbps":
                    row["downlink_throughput_mbps"],
                "uplink_throughput_mbps":
                    row["uplink_throughput_mbps"],
                "signal_strength_dbm":
                    row["signal_strength_dbm"],
                "active_connections":
                    row["active_connections"],
            },
        }

    finally:
        connection.close()


async def get_regional_network_summary(
    region: str,
) -> dict:
    """
    Get the current network health summary for a region.

    Reads all towers in the requested region and uses only
    the latest performance sample for each tower.
    """

    if not DB_PATH.exists():
        return {
            "success": False,
            "error": f"Database not found: {DB_PATH}",
        }

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            WITH latest_samples AS (
                SELECT
                    tower_id,
                    MAX(recorded_at) AS latest_recorded_at
                FROM tower_performance
                GROUP BY tower_id
            )

            SELECT
                t.tower_id,
                t.tower_name,
                t.region,
                t.city,
                t.state,
                t.technology,
                t.status,

                p.recorded_at,
                p.latency_ms,
                p.packet_loss_pct,
                p.downlink_throughput_mbps,
                p.uplink_throughput_mbps,
                p.signal_strength_dbm,
                p.active_connections

            FROM network_towers AS t

            LEFT JOIN latest_samples AS latest
                ON latest.tower_id = t.tower_id

            LEFT JOIN tower_performance AS p
                ON p.tower_id = latest.tower_id
                AND p.recorded_at = latest.latest_recorded_at

            WHERE LOWER(t.region) = LOWER(?)

            ORDER BY t.tower_id
            """,
            (region,),
        )

        rows = cursor.fetchall()

        if not rows:
            return {
                "success": False,
                "region": region,
                "error": "No towers found for this region.",
            }

        status_counts = {}
        towers = []

        for row in rows:
            status = row["status"]

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            towers.append(
                {
                    "tower_id": row["tower_id"],
                    "tower_name": row["tower_name"],
                    "city": row["city"],
                    "state": row["state"],
                    "technology": row["technology"],
                    "status": status,
                    "latest_performance": {
                        "recorded_at": row["recorded_at"],
                        "latency_ms": row["latency_ms"],
                        "packet_loss_pct":
                            row["packet_loss_pct"],
                        "downlink_throughput_mbps":
                            row["downlink_throughput_mbps"],
                        "uplink_throughput_mbps":
                            row["uplink_throughput_mbps"],
                        "signal_strength_dbm":
                            row["signal_strength_dbm"],
                        "active_connections":
                            row["active_connections"],
                    },
                }
            )

        return {
            "success": True,
            "region": region,
            "total_towers": len(rows),
            "status_counts": status_counts,
            "towers": towers,
        }

    finally:
        connection.close()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY was not found in .env"
    )


claude_model = LiteLlm(
    model=f"anthropic/{ANTHROPIC_MODEL}"
)


root_agent = Agent(
    name="network_diagnostics_agent",

    model=claude_model,

    description=(
        "Network Operations Center specialist that diagnoses "
        "telecom tower, connectivity, and regional network issues."
    ),

    instruction="""
You are the Network Diagnostics specialist for a telecom
Network Operations Center.

Your responsibility is to investigate tower health,
connectivity problems, active network incidents, and
regional network conditions.

You have three SQL-backed tools.

Use check_tower_status when the user asks about the current
status, latest performance, or active incidents for a
specific tower.

Use run_connectivity_diagnostics when the user reports a
connectivity symptom such as dropped sessions, packet loss,
high latency, slow throughput, weak signal, or unstable
service near a specific tower.

Use get_regional_network_summary when the user asks about
network health across an entire region.

Rules:

- Always use the available tools for operational network data.
- Never invent tower status or performance measurements.
- Never invent incidents.
- Never invent regional network information.
- Treat the SQL-backed tool results as the source of truth.
- For current network performance, use the latest sample
  returned by the tools.
- Distinguish degraded performance from a complete outage.
- Do not describe an OPERATIONAL tower as OFFLINE.
- Explain the diagnostic evidence clearly.
- Mention relevant packet loss, latency, throughput,
  signal strength, tower status, and incidents when available.
- If the data does not support a conclusion, say so.
""",

    tools=[
        check_tower_status,
        run_connectivity_diagnostics,
        get_regional_network_summary,
    ],
)


a2a_app = to_a2a(
    root_agent,
    port=8001,
)