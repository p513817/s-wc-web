import logging

import streamlit as st

from swc import handlers
from swc.handlers import config

logger = logging.getLogger(__name__)

## Shared
CFG = "config"
## SSD
SSD_MODE_RAD = "ssd_mode_radio"
M_MOCK, M_DET = "mock", "detect"
SSD_MODE_OPTS, SSD_MODE_CAPS = (
    [M_DET, M_MOCK],
    ["Detect SSD Automatically", "Mock SSD Name"],
)
SSD_MOCK_INP = "ssd_mode_input"
SSD_ISMART_PATH = "ssd_ismart_path"

## AIDA
AIDA_ENABLE_CBX = "aida_enable_checkbox"
AIDA_EXEC_PATH_INP = "aida_exec_path_input"
AIDA_OUT_DIR_INP = "aida_out_dir_input"
AIDA_DIR_KW_INP = "aida_dir_kw_input"

## IVIT
IVIT_ENABLE_CBX = "ivit_enable_checkbox"
IVIT_MODE_RAD = "ivit_mode_radio"
M_GEN, M_VAL = "generatic", "validator"
IVIT_MODE_OPTS, IVIT_MODE_CAPS = (
    [M_GEN, M_VAL],
    ["Generatic Mode Must Setup 2 Model", "Validator Mode Only Need One"],
)
IVIT_FROM_CSV_CBX = "ivit_from_csv_checkbox"
IVIT_IMAGE_DIR_INP = "ivit_image_dir_input"
IVIT_RULEBASE_CBX = "ivit_rulebase_checkbox"

## IVIT Model
R, W = "read", "write"
IVIT_TRG_MDL_OPTS = [R, W]
IVIT_TRG_MDL_SEL = "ivit_target_model_selectbox"
IVIT_RMDL_DIR_INP = "ivit_read_model_dir_inp"
IVIT_WMDL_DIR_INP = "ivit_write_model_dir_inp"
IVIT_RMDL_THRES = "ivit_read_model_threshold"
IVIT_WMDL_THRES = "ivit_write_model_threshold"

## Output
RET_DIR_INP = "retrain_dir_inp"
CUR_DIR_INP = "current_dir_inp"
HIS_DIR_INP = "history_dir_inp"

## Debug
DEBUG_SSD_ENABLE_CBX = "debug_ssd_enable_checkbox"
DEBUG_AIDA_ENABLE_CBX = "debug_aida_enable_checkbox"


def add_ssd_section(session, header: str = "SSD") -> None:
    cfg: config.Config = session[CFG]

    st.subheader(header)
    st.radio(
        "Select SSD Mode",
        options=SSD_MODE_OPTS,
        captions=SSD_MODE_CAPS,
        key=SSD_MODE_RAD,
        index=SSD_MODE_OPTS.index(cfg.ssd.mode),
        label_visibility="collapsed",
        format_func=handlers.st_formatter.upper_format_func,
    )
    if session[SSD_MODE_RAD] == M_MOCK:
        st.text_input(
            label="SSD Name",
            key=SSD_MOCK_INP,
            value=cfg.ssd.mock_name,
            placeholder="Innodisk 3TEA",
        )
    else:
        st.text_input(
            label="iSMART Executable File Path (*.exe)",
            key=SSD_ISMART_PATH,
            value=cfg.ssd.ismart_path,
            placeholder=r"C:\Users\DQE\Desktop\iSMART_6.4.18\iSMART.exe",
            help="Enter the absolute path to iSMART executable file",
        )


def add_aida_section(session, header: str = "AIDA64") -> None:
    cfg: config.Config = session[CFG]

    st.subheader(header)
    st.checkbox(
        label=f"Enable {header}",
        value=cfg.aida.enable,
        label_visibility="visible",
        disabled=session[SSD_MODE_RAD] == M_MOCK,
        key=AIDA_ENABLE_CBX,
    )
    if session[SSD_MODE_RAD] == M_DET and session[AIDA_ENABLE_CBX]:
        st.text_input(
            label="Executable file path",
            value=cfg.aida.exec_path,
            key=AIDA_EXEC_PATH_INP,
        )
        st.text_input(
            label="Generate file folder",
            value=cfg.aida.out_dir,
            key=AIDA_OUT_DIR_INP,
        )
        st.text_input(
            label="Generate file folder keyword",
            value=cfg.aida.dir_kw,
            key=AIDA_DIR_KW_INP,
        )


def add_ivit_section(session, header: str = "IVIT") -> None:
    cfg: config.Config = session[CFG]
    st.subheader(header)

    # enable
    st.checkbox(
        label=f"Enable {header}",
        value=cfg.ivit.enable,
        label_visibility="visible",
        key=IVIT_ENABLE_CBX,
    )
    if not session[IVIT_ENABLE_CBX]:
        return

    # input image directory
    if session[SSD_MODE_RAD] == "mock" or not session[AIDA_ENABLE_CBX]:
        st.text_input(
            label="Input Image / CSV directory",
            key=IVIT_IMAGE_DIR_INP,
            value=cfg.ivit.input_dir,
        )

    # from csv
    st.checkbox(
        label="Data From CSV",
        value=cfg.ivit.from_csv,
        label_visibility="visible",
        key=IVIT_FROM_CSV_CBX,
    )

    # rulebase
    st.checkbox(
        label="Rulebase Validation",
        value=cfg.ivit.rulebase if session[IVIT_FROM_CSV_CBX] == True else False,
        label_visibility="visible",
        key=IVIT_RULEBASE_CBX,
        disabled=not session[IVIT_FROM_CSV_CBX],
    )

    # ivit mode
    correct_opts, correct_caps = IVIT_MODE_OPTS, IVIT_MODE_CAPS
    if session[SSD_MODE_RAD] == "mock":
        correct_opts.pop(0)
        correct_caps.pop(0)

    st.radio(
        "Select IVIT Mode",
        options=correct_opts,
        captions=correct_caps,
        key=IVIT_MODE_RAD,
        index=correct_opts.index(cfg.ivit.mode)
        if cfg.ivit.mode and cfg.ivit.mode in correct_opts
        else 0,
        label_visibility="collapsed",
    )
    if session[IVIT_MODE_RAD] == M_VAL:
        st.selectbox(
            label="Select the target model",
            options=IVIT_TRG_MDL_OPTS,
            key=IVIT_TRG_MDL_SEL,
            index=IVIT_TRG_MDL_OPTS.index(cfg.ivit.target_model)
            if cfg.ivit.target_model
            else 0,
            label_visibility="collapsed",
        )

    # ivit model
    rw_model_flag = {R: False, W: False}
    if session[IVIT_MODE_RAD] == M_GEN:
        rw_model_flag[R], rw_model_flag[W] = True, True
    else:
        rw_model_flag[session[IVIT_TRG_MDL_SEL]] = True

    if rw_model_flag[R]:
        st.text_input(
            label=f"IVIT Model Directory :: {R.upper()}",
            value=cfg.ivit.models.read.model_dir,
            key=IVIT_RMDL_DIR_INP,
        )
        st.slider(
            label="Threshold", value=cfg.ivit.models.read.thres, key=IVIT_RMDL_THRES
        )
    if rw_model_flag[W]:
        st.text_input(
            label=f"IVIT Model Directory :: {W.upper()}",
            value=cfg.ivit.models.write.model_dir,
            key=IVIT_WMDL_DIR_INP,
        )
        st.slider(
            label="Threshold", value=cfg.ivit.models.write.thres, key=IVIT_WMDL_THRES
        )


def add_output_section(session, header: str = "OUTPUT") -> None:
    cfg: config.Config = session[CFG]
    st.subheader(header)
    st.text_input(
        label="Enter the retrain directory", key=RET_DIR_INP, value=cfg.output.retrain
    )
    st.text_input(
        label="Enter the current directory", key=CUR_DIR_INP, value=cfg.output.current
    )
    st.text_input(
        label="Enter the history directory", key=HIS_DIR_INP, value=cfg.output.history
    )


def add_debug_section(session, header: str = "DEBUG") -> None:
    cfg: config.Config = session[CFG]
    st.subheader(header)
    # enable
    st.checkbox(
        label="Mock SSD function for debug",
        value=cfg.debug.mock_ssd_process,
        label_visibility="visible",
        key=DEBUG_SSD_ENABLE_CBX,
    )
    st.checkbox(
        label="Mock AIDA function for debug",
        value=cfg.debug.mock_aida_process,
        label_visibility="visible",
        key=DEBUG_AIDA_ENABLE_CBX,
    )


def main(session):
    st.title("Setting")
    add_ssd_section(session)
    add_aida_section(session)
    add_ivit_section(session)
    add_output_section(session)
    add_debug_section(session)

    if st.button("Submit", use_container_width=True, type="primary"):
        logger.info("Click submit button")
        cfg = config.Config(
            ssd=config.SSD(
                mode=session[SSD_MODE_RAD],
                mock_name=session.get(SSD_MOCK_INP, ""),
                ismart_path=session.get(SSD_ISMART_PATH, None),
            ),
            aida=config.AIDA(
                enable=session[SSD_MODE_RAD] == "detect" and session[AIDA_ENABLE_CBX],
                exec_path=session.get(AIDA_EXEC_PATH_INP, ""),
                out_dir=session.get(AIDA_OUT_DIR_INP, ""),
                dir_kw=session.get(AIDA_DIR_KW_INP, ""),
            ),
            ivit=config.IVIT(
                enable=session[IVIT_ENABLE_CBX],
                mode=session.get(IVIT_MODE_RAD),
                target_model=session.get(IVIT_TRG_MDL_SEL),
                from_csv=session.get(IVIT_FROM_CSV_CBX),
                rulebase=session.get(IVIT_RULEBASE_CBX)
                if session.get(IVIT_FROM_CSV_CBX)
                else False,
                input_dir=session.get(IVIT_IMAGE_DIR_INP),
                models=config.IVIT_RW_Model(
                    read=config.IVITModel(
                        model_dir=session.get(IVIT_RMDL_DIR_INP),
                        thres=session.get(IVIT_RMDL_THRES, 0.1),
                    ),
                    write=config.IVITModel(
                        model_dir=session.get(IVIT_WMDL_DIR_INP),
                        thres=session.get(IVIT_WMDL_THRES, 0.1),
                    ),
                ),
            ),
            output=config.OUTPUT(
                retrain=session[RET_DIR_INP],
                current=session[CUR_DIR_INP],
                history=session[HIS_DIR_INP],
            ),
            debug=config.DEBUG(
                mock_aida_process=session[DEBUG_AIDA_ENABLE_CBX],
                mock_ssd_process=session[DEBUG_SSD_ENABLE_CBX],
            ),
        )
        logger.info("Initialized config object")
        # Validation and Save
        is_valid = False
        try:
            handlers.ssd.validation(cfg=cfg)
            if cfg.aida.enable:
                handlers.aida.validate(cfg=cfg)
            if cfg.ivit.enable:
                handlers.ivit.validate(cfg=cfg)
            config.save(cfg.model_dump())
            is_valid = True
        except Exception as e:
            st.exception(e)

        if is_valid:
            # put switch_page in try will cause error
            st.switch_page("runtime.py")


# ---------------------------------------------

session = st.session_state
if CFG not in session:
    session[CFG] = config.get()
main(session)
