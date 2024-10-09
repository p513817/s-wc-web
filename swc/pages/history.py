import logging
import pathlib

import handlers
import streamlit as st
from handlers import config, log
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

import swc

log.setup()
logger = logging.getLogger(__name__)

CFG = "config"
PG_ID = "page_id"
PG_SIZE = 10


@st.dialog("Log Content", width="large")
def log_txt_dialog(log_path: str):
    log_path = str(log_path)

    with open(log_path, "r") as f:
        lines = "".join([line.replace("\t", "  ") for line in f.readlines()])
    st.code(lines)


def go_next_page():
    session[PG_ID] = session[PG_ID] + PG_SIZE


def go_previous_page():
    session[PG_ID] = session[PG_ID] - PG_SIZE


def main(session: SessionStateProxy):
    """
    1. 運行所有的 validate
    2. 運行所有的 event
    """

    cfg: handlers.config.Config = session[CFG]
    # Title
    st.title("S-WC", help=f"S-WC Version: {swc.__version__}")
    st.header("History")
    # Param
    if PG_ID not in session:
        session[PG_ID] = 0

    # Variable
    history_root = pathlib.Path(cfg.output.history)
    history_paths = sorted(history_root.glob("*.log"), reverse=True)
    history_length = len(history_paths)
    page_id = session[PG_ID] // PG_SIZE + 1
    pages = history_length // PG_SIZE + 1

    st.subheader(f"Logs ({page_id}/{pages})", divider=True)

    log_cntr = st.columns([1, 1])
    start, end = session[PG_ID], session[PG_ID] + PG_SIZE
    end = end if end < history_length else history_length
    for idx in range(start, end):
        history_path = history_paths[idx]
        cntr_id = 0 if idx % PG_SIZE < 5 else 1
        with log_cntr[cntr_id]:
            st.button(
                label=history_path.name,
                on_click=log_txt_dialog,
                args=(history_path,),
                use_container_width=True,
            )
    bt_cntr = st.columns([1, 1])
    with bt_cntr[0]:
        st.button(
            label="Previous",
            use_container_width=True,
            on_click=go_previous_page,
            disabled=session[PG_ID] == 0,
        )
    with bt_cntr[1]:
        st.button(
            label="Next",
            use_container_width=True,
            type="primary",
            on_click=go_next_page,
            disabled=session[PG_ID] + PG_SIZE >= history_length,
        )


# Main
session = st.session_state
session[CFG] = cfg = config.get()
main(session)
