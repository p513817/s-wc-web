import logging
from typing import List

import handlers
import pandas as pd
import plotly.express as px
import streamlit as st
from handlers import config, log
from streamlit.runtime.state.session_state_proxy import SessionStateProxy

log.setup()
logger = logging.getLogger(__name__)

SSD_DETECT_SEL = "ssd_detect_selectbox"
KW_R = "_R"
KW_W = "_W"

# --------------------------------------------------


def ssd_event(session: SessionStateProxy):
    """
    1. 如果是 mock 模式就直接顯示設定的 disk
    2. 如果是 detect 模式就要進行偵測
    """
    logger.info("Run SSD Event")

    cfg: config.Config = session[CFG]
    if cfg.ssd.mode == "mock":
        st.info(f"Mock SSD: {cfg.ssd.mock_name}")

    elif cfg.ssd.mode == "detect":
        detect_ssd_func = (
            handlers.ssd.mock_detect
            if cfg.debug.mock_ssd_process
            else handlers.ssd.detect
        )
        disks = detect_ssd_func(ismart_path=cfg.ssd.ismart_path)

        handlers.ssd.valid_detected_disks(disks=disks)
        st.selectbox(label="Select SSD", options=disks, index=0, key=SSD_DETECT_SEL)


def aida_event(session: SessionStateProxy):
    cfg: config.Config = session[CFG]
    if not cfg.aida.enable:
        return

    logger.info("Run AIDA Event")
    # Run AIDA
    with st.spinner("Running AIDA64"):
        if cfg.debug.mock_aida_process:
            handlers.aida.mock_run_cmd(output_dir=cfg.aida.out_dir)
        else:
            handlers.aida.run_cmd(aida_exec_path=cfg.aida.exec_path)
    # Verify Folder
    gen_folders = handlers.aida.get_output_dir(cfg.aida.out_dir, cfg.aida.dir_kw)
    handlers.aida.valid_aida_output_folder(gen_folders)
    # Parse Folder
    images, csvs = handlers.aida.parse_aida_output_dir(
        gen_folders[0], cfg.ivit.mode == "validator"
    )
    st.success("AIDA Finished")


# --------------------------------------------------


def get_model_info(model_cfg: config.IVITModel):
    mdl_info = handlers.ivit.parse_ivit_model_dir(model_cfg.model_dir)
    mdl_info.threshold = model_cfg.thres
    return mdl_info


def ivit_generatic_event(session: SessionStateProxy) -> List[handlers.ivit.InferData]:
    cfg: config.Config = session[CFG]
    # Load Model
    rmdl_info = get_model_info(cfg.ivit.models.read)
    rmdl = handlers.ivit.get_model(_info=rmdl_info)

    wmdl_info = get_model_info(cfg.ivit.models.write)
    wmdl = handlers.ivit.get_model(_info=wmdl_info)

    # Load and Check Data
    read_inputs, write_inputs = handlers.aida.get_data(session[CFG])

    read_outputs = handlers.ivit.do_inference(
        model=rmdl, infer_data_list=read_inputs, from_csv=cfg.ivit.from_csv
    )
    write_outputs = handlers.ivit.do_inference(
        model=wmdl, infer_data_list=write_inputs, from_csv=cfg.ivit.from_csv
    )

    return read_outputs + write_outputs


def ivit_validator_event(session: SessionStateProxy) -> List[handlers.ivit.InferData]:
    st.info("Validator Mode")
    cfg: config.Config = session[CFG]

    # Get Data and Model Config
    read_data, write_data = handlers.aida.get_data(session[CFG])
    if cfg.ivit.target_model == "read":
        target_data = read_data
        model_cfg = cfg.ivit.models.read
    else:
        target_data = write_data
        model_cfg = cfg.ivit.models.write

    # Load Model
    tmdl_info = get_model_info(model_cfg=model_cfg)
    tmdl = handlers.ivit.get_model(_info=tmdl_info)

    # Do Inference
    target_outputs = handlers.ivit.do_inference(
        model=tmdl, infer_data_list=target_data, from_csv=cfg.ivit.from_csv
    )
    return target_outputs


def ivit_event(session: SessionStateProxy) -> List[handlers.ivit.InferData]:
    cfg: config.Config = session[CFG]
    if not cfg.ivit.enable:
        return []

    if cfg.ivit.mode == "generatic":
        return ivit_generatic_event(session)
    else:
        return ivit_validator_event(session)


# --------------------------------------------------


def get_report_plot(reports):
    # Start Report Event
    ret = []
    for report in reports:
        # Get Top 1 result
        data = report.data
        top1 = data.output[0]
        # For Plot
        ret.append(
            {
                "status": report.status,
                "data_path": handlers.reporter.image_to_base64(data.input.data_path),
                "plot_path": handlers.reporter.image_to_base64(data.input.plot_path),
                "verify_path": handlers.reporter.image_to_base64(
                    data.input.verify_path
                ),
                "domain": data.input.domain,
                "ai_verify": report.ai_verify,
                "rule_verify": report.rule_verify,
                "ground_truth": report.ground_truth,
                "label": top1.label,
                "confidence": top1.confidence,
            }
        )
    df = pd.DataFrame(ret)
    st.dataframe(
        df,
        column_config={
            "data_path": st.column_config.ImageColumn("Data Image", width="small"),
            "plot_path": st.column_config.ImageColumn("Plot Image", width="small"),
            "verify_path": st.column_config.ImageColumn("Verify Image", width="small"),
        },
        use_container_width=True,
    )
    status_counts = df["status"].value_counts()

    # Status 統計: {'PASS': 14, 'FAIL': 3}
    # st.write(f"Status 統計: {status_counts.to_dict()}")
    # Plot Pie Chart with Plotly
    fig = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        title="Status Distribution",
        labels={"names": "Status"},
    )

    # Show pie chart in Streamlit
    st.plotly_chart(fig)


def generatic_report_event(
    session: SessionStateProxy, infer_outputs: List[handlers.ivit.InferData]
):
    logger.info("Start Generatice Report")
    # Get Basic Information
    cfg: config.Config = session[CFG]
    if cfg.ssd.mode == "detect":
        ground_truth = session[SSD_DETECT_SEL]
    else:
        ground_truth = cfg.ssd.mock_name
    timestamp = handlers.reporter.get_timetamp()
    model_info_map = {
        "read": get_model_info(cfg.ivit.models.read),
        "write": get_model_info(cfg.ivit.models.write),
    }

    def report_wrapper(infer_data):
        return handlers.reporter.get_report(
            infer_data=[infer_data],
            ground_truth=ground_truth,
            model_info=model_info_map.get(infer_data.input.domain, None),
            config_info=cfg,
            timestamp=timestamp,
        )[0]

    # Get Report
    reports = list(map(report_wrapper, infer_outputs))
    # Process Report
    handlers.reporter.process(reports=reports)

    # Gen Plot
    get_report_plot(reports=reports)
    logger.info("Finished Generatice Report")


def validator_report_event(
    session: SessionStateProxy, infer_outputs: List[handlers.ivit.InferData]
):
    logger.info("Start Validator Report")
    logger.info("Finished Validator Report")
    if cfg.ssd.mode == "detect":
        ground_truth = session[SSD_DETECT_SEL]
    else:
        ground_truth = cfg.ssd.mock_name
    timestamp = handlers.reporter.get_timetamp()

    def report_wrapper(infer_data):
        return handlers.reporter.get_report(
            infer_data=[infer_data],
            ground_truth=ground_truth,
            model_info=None,
            config_info=cfg,
            timestamp=timestamp,
        )[0]

    # Get Report
    reports = list(map(report_wrapper, infer_outputs))

    # Save to CSV
    handlers.reporter.process_xml(reports=reports)

    get_report_plot(reports=reports)


def none_ivit_report_event(session: SessionStateProxy):
    logger.info("Start None iVIT Report")
    cfg: config.Config = session[CFG]
    read_inputs, write_inputs = handlers.aida.get_data(session[CFG])
    if cfg.ssd.mode == "detect":
        ground_truth = session[SSD_DETECT_SEL]
    else:
        ground_truth = cfg.ssd.mock_name
    timestamp = handlers.reporter.get_timetamp()

    logger.debug(read_inputs)
    logger.debug(write_inputs)

    # Gen InferData
    infer_outputs = [
        handlers.ivit.InferData(
            input=infer_input,
            output=[handlers.ivit.InferOutput(label=None, index=None, confidence=None)],
        )
        for infer_input in read_inputs + write_inputs
    ]

    def report_wrapper(infer_data):
        return handlers.reporter.get_report(
            infer_data=[infer_data],
            ground_truth=ground_truth,
            model_info=None,
            config_info=cfg,
            timestamp=timestamp,
        )[0]

    # Get Report
    reports = list(map(report_wrapper, infer_outputs))
    # Process Report
    handlers.reporter.process(reports=reports)
    get_report_plot(reports=reports)

    logger.info("Finished None iVIT Report")

    pass


def report_event(session: SessionStateProxy, infer_outputs=None):
    if cfg.ivit.enable:
        if cfg.ivit.mode == "generatic":
            return generatic_report_event(session, infer_outputs)
        else:
            return validator_report_event(session, infer_outputs)
    else:
        return none_ivit_report_event(session)


def main(session: SessionStateProxy):
    """
    1. 運行所有的 validate
    2. 運行所有的 event
    """

    cfg: handlers.config.Config = session[CFG]

    # Title
    st.title("Runtime")

    # Validate Each Feature
    is_valid = False
    try:
        handlers.ssd.validation(cfg=cfg)
        handlers.aida.validate(cfg=cfg)
        handlers.ivit.validate(cfg=cfg)
        is_valid = True
    except Exception as e:
        st.exception(e)

    if not is_valid:
        return

    # Run Each Feature
    try:
        with st.spinner("Detecting SSD ..."):
            ssd_event(session)

        bt_is_available = False
        if cfg.ssd.mode == "detect":
            bt_is_available = bool(session[SSD_DETECT_SEL] is not None)
        else:
            bt_is_available = True

        logger.info(f"Start button is : {bt_is_available}")

        if st.button(
            label="Start",
            use_container_width=True,
            type="primary",
            disabled=not bt_is_available,
        ):
            aida_event(session)
            infer_outputs = ivit_event(session)

            report_event(session, infer_outputs)

    except Exception as e:
        logger.exception(e)
        raise


# Main
CFG = "config"
session = st.session_state
session[CFG] = cfg = config.get()
main(session)
