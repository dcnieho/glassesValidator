import aiosqlite
import asyncio
import typing
import enum
import json
import dataclasses
import pathlib
import re
from concurrent.futures import Future

from glassesTools.eyetracker import EyeTracker
from glassesTools.recording import Recording as base_rec
from glassesTools.utils import hex_to_rgba_0_1, rgba_0_1_to_hex
from glassesTools import async_thread

from .structs import DefaultStyleDark, Recording, Settings
from . import globals, utils
from ...utils import Task

connection: aiosqlite.Connection = None
db_save_future: Future = None

def setup():
    global db_save_future

    async_thread.wait(connect())
    async_thread.wait(load())
    db_save_future = async_thread.run(save_loop())


def shutdown():
    global connection

    if db_save_future is not None:
        db_save_future.cancel()
    async_thread.wait(close())
    connection = None
    globals.rec_id.set_count(0)


async def create_table(table_name: str, columns: dict[str, str]):
    await connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join([f'{column_name} {column_def}' for column_name, column_def in columns.items()])}
        )
    """)

    # Add missing and update existing columns for backwards compatibility
    cursor = await connection.execute(f"""
        PRAGMA table_info({table_name})
    """)
    key_column = None
    for column_name, column_def in columns.items():
        if "primary key" in column_def.lower():
            key_column = column_name
            break
    has_columns = [tuple(row) for row in await cursor.fetchall()]  # (index, name, type, can_be_null, default, idk)
    has_column_names = [column[1] for column in has_columns]
    has_column_defs = [(column[2], column[4]) for column in has_columns]  # (type, default)
    for column_name, column_def in columns.items():
        if column_name not in has_column_names:
            # Column is missing, add it
            await connection.execute(f"""
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_def}
            """)
        elif key_column is not None:  # Can only attempt default fix if key is present to transfer values
            has_column_def = has_column_defs[has_column_names.index(column_name)]  # (type, default)
            if not column_def.strip().lower().startswith(has_column_def[0].lower()):
                raise Exception(f"Existing database column '{column_name}' has incorrect type ({column_def[:column_def.find(' ')]} != {has_column_def[0]})")
            if " default " in column_def.lower() and not re.search(r"[Dd][Ee][Ff][Aa][Uu][Ll][Tt]\s+?" + re.escape(str(has_column_def[1])), column_def):
                # Default is different, recreate column and transfer values
                cursor = await connection.execute(f"""
                    SELECT {key_column}, {column_name}
                    FROM {table_name}
                """)
                rows = await cursor.fetchall()
                await connection.execute(f"""
                    ALTER TABLE {table_name}
                    DROP COLUMN {column_name}
                """)
                await connection.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN {column_name} {column_def}
                """)
                for row in rows:
                    await connection.execute(f"""
                        UPDATE {table_name}
                        SET
                            {column_name} = ?
                        WHERE {key_column}=?
                    """, (row[column_name], row[key_column]))


async def connect():
    global connection

    data_path = utils.get_data_path()

    connection = await aiosqlite.connect(data_path / "db.sqlite3")
    connection.row_factory = aiosqlite.Row  # Return sqlite3.Row instead of tuple

    await create_table("settings", {
        "_":                           f'INTEGER PRIMARY KEY CHECK (_=0)',
        "config_dir":                  f'INTEGER DEFAULT "config"',
        "confirm_on_remove":           f'INTEGER DEFAULT {int(True)}',
        "continue_process_after_code": f'INTEGER DEFAULT {int(True)}',
        "copy_scene_video":            f'INTEGER DEFAULT {int(True)}',
        "dq_use_viewpos_vidpos_homography": f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_vidpos_homography":    f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_vidpos_ray":      f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_world_eye":       f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_left_eye":        f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_right_eye":       f'INTEGER DEFAULT {int(False)}',
        "dq_use_pose_left_right_avg":  f'INTEGER DEFAULT {int(False)}',
        "dq_report_data_loss":         f'INTEGER DEFAULT {int(False)}',
        "fix_assign_do_global_shift":  f'INTEGER DEFAULT {int(True)}',
        "fix_assign_max_dist_fac":     f'FLOAT DEFAULT .5',
        "process_workers":             f'INTEGER DEFAULT 2',
        "render_when_unfocused":       f'INTEGER DEFAULT {int(True)}',
        "show_advanced_options":       f'INTEGER DEFAULT {int(False)}',
        "show_remove_btn":             f'INTEGER DEFAULT {int(True)}',
        "style_accent":                f'TEXT    DEFAULT "{DefaultStyleDark.accent}"',
        "style_alt_bg":                f'TEXT    DEFAULT "{DefaultStyleDark.alt_bg}"',
        "style_bg":                    f'TEXT    DEFAULT "{DefaultStyleDark.bg}"',
        "style_border":                f'TEXT    DEFAULT "{DefaultStyleDark.border}"',
        "style_color_recording_name":  f'INTEGER DEFAULT {int(True)}',
        "style_corner_radius":         f'INTEGER DEFAULT {DefaultStyleDark.corner_radius}',
        "style_text":                  f'TEXT    DEFAULT "{DefaultStyleDark.text}"',
        "style_text_dim":              f'TEXT    DEFAULT "{DefaultStyleDark.text_dim}"',
        "vsync_ratio":                 f'INTEGER DEFAULT 1',
    })
    await connection.execute("""
        INSERT INTO settings
        (_)
        VALUES
        (0)
        ON CONFLICT DO NOTHING
    """)

    if globals.project_path is not None:
        await create_table("recordings", {
            "id":                           f'INTEGER PRIMARY KEY',
            "name":                         f'TEXT    DEFAULT ""',
            "source_directory":             f'TEXT    DEFAULT ""',
            "working_directory":            f'TEXT    DEFAULT ""',
            "start_time":                   f'INTEGER DEFAULT 0',
            "duration":                     f'INTEGER DEFAULT 0',
            "eye_tracker":                  f'TEXT    DEFAULT "{EyeTracker.Unknown.value}"',
            "eye_tracker_name":             f'TEXT    DEFAULT ""',
            "project":                      f'TEXT    DEFAULT ""',
            "participant":                  f'TEXT    DEFAULT ""',
            "firmware_version":             f'TEXT    DEFAULT ""',
            "glasses_serial":               f'TEXT    DEFAULT ""',
            "recording_unit_serial":        f'TEXT    DEFAULT ""',
            "recording_software_version":   f'TEXT    DEFAULT ""',
            "scene_camera_serial":          f'TEXT    DEFAULT ""',
            "scene_video_file":             f'TEXT    DEFAULT ""',
            "task":                         f'TEXT    DEFAULT "{Task.Unknown.value}"',
        })


async def save():
    await connection.commit()


async def save_loop():
    while True:
        await asyncio.sleep(30)
        await save()


async def close():
    if connection is not None:
        await save()
        await connection.close()


def sql_to_py(value: str | int | float, data_type: typing.Type):
    match getattr(data_type, "__name__", None):
        case "list":
            value = json.loads(value)
            if hasattr(data_type, "__args__"):
                content_type = data_type.__args__[0]
                value = [content_type(x) for x in value]
        case "tuple":
            if isinstance(value, str) and getattr(data_type, "__args__", [None])[0] is float:
                value = hex_to_rgba_0_1(value)
            else:
                value = json.loads(value)
                if hasattr(data_type, "__args__"):
                    content_type = data_type.__args__[0]
                    value = [content_type(x) for x in value]
                value = tuple(value)
        case "str":
            if value=='null':
                value = ""
            else:
                value = data_type(value)
        case "int":
            if value=='null':
                value = None
            else:
                value = data_type(value)
        case "float":
            if value=='null':
                value = None
            else:
                value = data_type(value)
        case "Timestamp":
            if value=='null':
                value = 0
            value = data_type(value)
        case _:
            value = data_type(value)
    return value


async def load_recordings(id: int = None):
    types = Recording.__annotations__|base_rec.__annotations__
    query = """
        SELECT *
        FROM recordings
    """
    if id is not None:
        query += f"""
            WHERE id={id}
        """
    cursor = await connection.execute(query)
    recordings = await cursor.fetchall()
    for recording in recordings:
        recording = dict(recording)
        recording = {key: sql_to_py(value, types[key]) for key, value in recording.items() if key in types}
        globals.recordings[recording["id"]] = Recording(**recording)
        globals.selected_recordings[recording["id"]] = False

        # check if working dir exists. If not, ensure recording status is not imported
        rec = globals.recordings[recording["id"]]
        if rec.working_directory and not rec.working_directory.is_dir():
            rec.task = Task.Not_Imported
            async_thread.run(update_recording(rec, "task"))


async def load():
    types = Settings.__annotations__
    cursor = await connection.execute("""
        SELECT *
        FROM settings
    """)
    settings = dict(await cursor.fetchone())
    settings = {key: sql_to_py(value, types[key]) for key, value in settings.items() if key in types}
    globals.settings = Settings(**settings)

    globals.recordings = {}
    globals.selected_recordings = {}
    if globals.project_path is not None:
        await load_recordings()
        if globals.recordings:
            globals.rec_id.set_count(max(globals.recordings.keys()))
        else:
            globals.rec_id.set_count(0)


def py_to_sql(value: enum.Enum | bool | list | tuple | typing.Any):
    if hasattr(value, "value"):
        value = value.value
    elif hasattr(value, "hash"):
        value = value.hash
    elif isinstance(value, pathlib.Path):
        value = str(value)
    elif isinstance(value, bool):
        value = int(value)
    elif isinstance(value, list):
        value = list(value)
        value = [item.value if hasattr(item, "value") else item for item in value]
        value = json.dumps(value)
    elif isinstance(value, tuple) and 3 <= len(value) <= 4:
        value = rgba_0_1_to_hex(value)
    return value


async def update_recording(recording: Recording, *keys: list[str]):
    values = []

    for key in keys:
        value = py_to_sql(getattr(recording, key))
        values.append(value)

    await connection.execute(f"""
        UPDATE recordings
        SET
            {", ".join(f"{key} = ?" for key in keys)}
        WHERE id={recording.id}
    """, tuple(values))


async def update_settings(*keys: list[str]):
    values = []

    for key in keys:
        value = py_to_sql(getattr(globals.settings, key))
        values.append(value)

    await connection.execute(f"""
        UPDATE settings
        SET
            {", ".join(f"{key} = ?" for key in keys)}
        WHERE _=0
    """, tuple(values))


async def remove_recording(id: int):
    await connection.execute(f"""
        DELETE FROM recordings
        WHERE id={id}
    """)


async def add_recording(rec: Recording):
    async with globals.rec_id:
        rec.id = globals.rec_id.get_count()

    keys    = ", ".join(f"{key.name}" for key in dataclasses.fields(rec))
    values  = "{}".format(tuple(py_to_sql(at) if (at:=getattr(rec, key.name)) else "null" for key in dataclasses.fields(rec)))

    await connection.execute(f"""
        INSERT INTO recordings
        ({keys})
        VALUES
        {values}
    """)

    return rec.id