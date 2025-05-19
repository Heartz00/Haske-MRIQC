import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st
import zipfile
import uuid
import shutil
import subprocess
import requests
from pathlib import Path
import os
import json
import datetime
import time
from io import BytesIO
# ------------------------------
# Streamlit Page Configuration & Branding
# ------------------------------
st.set_page_config(page_title="MRIQC App", layout="wide")

st.markdown("""
# MRIQC Web App for MRI Image Quality Assessment

The Web-MRIQC App provides an intuitive web interface for running Quality Control on MRI datasets acquired in DICOM formats. The App offers users the ability to compute Image Quality Metrics (IQMs) for neuroimaging studies.
This web-based solution implements the original MRIQC standalone application in a user-friendly interface accessible from any device, without the need for software installation or access to resource-intensive computers. Thus, simplifying the quality control workflow. For a comprehensive understanding of the IQMs computed by MRIQC, as well as details on the original MRIQC implementation, refer to the official MRIQC documentation: https://mriqc.readthedocs.io.
""",
            unsafe_allow_html=True
            )

st.markdown(
    """
## How to Use:
The app enables users to upload T1w, T2w, DWI, BOLD fMRI, or ASL DICOM files as a folder or zipped format, convert them to the Standard Brain Imaging Data Structure (BIDS) format using dcm2bids [1] via dcm2niiX [2], and then process the IQMs using MRIQC [3]. The resulting reports can be downloaded for further analysis. To use, follow the following steps:

1. Enter Subject ID (optional)
2. Enter the Session ID (optional, e.g, baseline, follow up, etc)
3. Select your preferred modality for analysis (T1w, T2w, DWI, BOLD fMRI, or ASL)
4. Upload a zipped file/folder containing T1w, T2w, DWI, BOLD fMRI, or ASL DICOM images by dragging and dropping the zipped file or uploading using the browse file option
5. Click DICOM → BIDS Conversion
6. Once BIDS converted, you will see the notification: DICOM to BIDS conversion complete
7. Click Send BIDS to Web for MRIQC or if you want the BIDS format, Click Download BIDS Dataset to your device.
8. Send the converted BIDS images to MRIQC by clicking Send BIDS to Web for MRIQC  for generating the IQMs
9. Depending on your internet connection, this can between 5-10 minutes to get your results for a single participant.
10. When completed, you can view the report on the web App or download the report of the IQM by clicking the "Download MRIQC results" button including the csv export.

## References
1. Boré, A., Guay, S., Bedetti, C., Meisler, S., & GuenTher, N. (2023). Dcm2Bids (Version 3.1.1) [Computer software]. https://doi.org/10.5281/zenodo.8436509
2. Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods., 2016, 264:47-56.
3. Esteban O, Birman D, Schaer M, Koyejo OO, Poldrack RA, Gorgolewski KJ (2017) MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites. PLoS ONE 12(9): e0184661. https://doi.org/10.1371/journal.pone.0184661
""", unsafe_allow_html=True)

# Display IQM tables in Markdown
st.markdown("""
### **Anatomical (T1w / T2w) IQMs**

| Abbreviation | Name                                 | Description                                                                                                                                    |
|--------------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| **CNR**      | Contrast-to-Noise Ratio              | Measures how well different tissues (like gray matter and white matter) are distinguished. Higher CNR indicates better tissue contrast.        |
| **SNR**      | Signal-to-Noise Ratio                | Assesses the strength of the signal relative to background noise. Higher SNR means clearer images.                                             |
| **EFC**      | Entropy Focus Criterion              | Quantifies image sharpness using Shannon entropy. Higher EFC indicates more ghosting/blurring (i.e., less sharp).                              |
| **FBER**     | Foreground-Background Energy Ratio   | Compares energy inside the brain mask vs outside. Higher FBER reflects better tissue delineation.                                              |
| **FWHM**     | Full Width at Half Maximum           | Estimates the smoothness in spatial resolution. Lower FWHM typically implies sharper images (depends on scanner/protocol).                     |
| **INU**      | Intensity Non-Uniformity             | Evaluates bias fields caused by scanner imperfections. Higher INU suggests more uneven signal across the image.                                |
| **Art_QI1**  | Quality Index 1                      | Measures artifacts in areas outside the brain. Higher QI1 = more artifacts (e.g., motion, ghosting).                                           |
| **Art_QI2**  | Quality Index 2                      | Detects structured noise using a chi-squared goodness-of-fit test. Higher QI2 indicates potential issues with signal consistency.              |
| **WM2MAX**   | White Matter to Max Intensity Ratio  | Checks if white matter intensity is within a normal range. Very high or low values may indicate problems with normalization or acquisition.    |

### **Functional (BOLD MRI) IQMs**

| Abbreviation | Name                               | Description                                                                                                                    |
|--------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **FD**       | Framewise Displacement             | Quantifies subject head movement across volumes. Higher FD = more motion artifacts. Mean FD < 0.2 mm is often acceptable.     |
| **DVARS**    | D Temporal Variance of Signal      | Measures the change in signal between consecutive volumes. Spikes in DVARS can indicate motion or noise events.               |
| **tSNR**     | Temporal Signal-to-Noise Ratio     | Assesses the SNR over time (mean / std of the time series per voxel). Higher tSNR = more reliable signal over time.           |
| **GCOR**     | Global Correlation                 | Detects global signal fluctuations across the brain. Elevated GCOR may reflect widespread noise.                              |
| **AOR**      | AFNI Outlier Ratio                 | Counts the number of voxels flagged as statistical outliers. High AOR suggests poor scan quality or significant motion issues. |
| **GSR**      | Global Signal Regression Impact    | Assesses how removing global signal changes BOLD contrast. Large differences might affect downstream analysis.                |

*For deeper technical explanations, see the [MRIQC Documentation](https://mriqc.readthedocs.io/en/latest/iqms/iqms.html).*
""")

# ------------------------------
# Helper Functions
# ------------------------------


def generate_dcm2bids_config(temp_dir: Path) -> Path:
    config = {
        "descriptions": [
            # Anatomical Imaging
            {
                "datatype": "anat",
                "suffix": "T1w",
                "criteria": {
                    "SeriesDescription": "*T1*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|PERMANY|OTHER).*"]
                },
                "sidecar_changes": {"ProtocolName": "T1w"}
            },
            {
                "datatype": "anat",
                "suffix": "T2w",
                "criteria": {
                    "SeriesDescription": "*T2*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|PERMANY).*"]
                },
                "sidecar_changes": {"ProtocolName": "T2w"}
            },
            {
                "datatype": "anat",
                "suffix": "FLAIR",
                "criteria": {
                    "SeriesDescription": "*FLAIR*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|PERMANY).*"]
                }
            },

            # Functional Imaging
            {
                "datatype": "func",
                "suffix": "bold",
                "criteria": {
                    "SeriesDescription": "*BOLD*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|FMRI|OTHER).*"]
                },
                "sidecar_changes": {"TaskName": "rest"}
            },
            {
                "datatype": "func",
                "suffix": "sbref",
                "criteria": {
                    "SeriesDescription": "*SBRef*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|FMRI|OTHER).*"]
                }
            },

            # Diffusion Imaging
            {
                "datatype": "dwi",
                "suffix": "dwi",
                "criteria": {
                    "SeriesDescription": "*DWI*|*DTI*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|DIFFUSION).*"]
                },
                "sidecar_changes": {
                    "PhaseEncodingDirection": "j",
                    "TotalReadoutTime": 0.028
                }
            },

            # Field Maps
            {
                "datatype": "fmap",
                "suffix": "phasediff",
                "criteria": {
                    "SeriesDescription": "*FMRI_DISTORTION*",
                    "ImageType": ["ORIGINAL", "(?i).*(P|PHASE).*"]
                }
            },
            {
                "datatype": "fmap",
                "suffix": "magnitude",
                "criteria": {
                    "SeriesDescription": "*FMRI_DISTORTION*",
                    "ImageType": ["ORIGINAL", "(?i).*(M|MAG).*"]
                }
            },

            # Perfusion Imaging
            {
                "datatype": "perf",
                "suffix": "asl",
                "criteria": {
                    "SeriesDescription": "*ASL*|*Perfusion*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|PERFUSION).*"]
                }
            },

            # Task-Based fMRI(Example for different tasks)
            {
                "datatype": "func",
                "suffix": "bold",
                "criteria": {
                    "SeriesDescription": "*Nback*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|FMRI).*"]
                },
                "sidecar_changes": {"TaskName": "nback"}
            },

            # Multi-echo Sequences
            {
                "datatype": "anat",
                "suffix": "MESE",
                "criteria": {
                    "SeriesDescription": "*MultiEcho*",
                    "ImageType": ["ORIGINAL", "(?i).*(PRIMARY|MULTIECHO).*"]
                }
            }
        ],
        "default_entities": {
            "subject": "{subject}",
            "session": "{session}"
        }
    }
    config_file = temp_dir / "dcm2bids_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    return config_file


def run_dcm2bids(dicom_dir: Path, bids_out: Path, subj_id: str, ses_id: str, config_file: Path):
    cmd = ["dcm2bids", "-d", str(dicom_dir), "-p", subj_id,
           "-c", str(config_file), "-o", str(bids_out)]
    if ses_id:
        cmd += ["-s", ses_id]
    st.write(f"**Running**: `{' '.join(cmd)}`")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"dcm2bids error:\n{result.stderr}")
    else:
        st.success("dcm2bids completed successfully.")


def classify_from_metadata(meta):
    """
    Classifies based on metadata if and only if ImageType includes 'ORIGINAL'.
    """
    image_type = meta.get("ImageType", [])
    if isinstance(image_type, str):
        image_type = [image_type]

    if not any("original" in t.lower() for t in image_type):
        return None, None  # Skip derived images

    desc = (meta.get("SeriesDescription", "") + " " +
            meta.get("ProtocolName", "")).lower()
    pulse = meta.get("PulseSequenceName", "").lower()

    if "t1" in desc and "flair" not in desc:
        return "anat", "T1w"
    elif "t2" in desc:
        return "anat", "T2w"
    elif "flair" in desc or "fluid" in desc:
        return "anat", "FLAIR"
    elif "dwi" in desc or "dti" in desc:
        return "dwi", "dwi"
    elif "bold" in desc or "fmri" in desc or "functional" in desc or "activation" in desc or "epi" in pulse:
        return "func", "bold"
    elif "asl" in desc or "perfusion" in desc:
        return "perf", "asl"
    else:
        return None, None


def classify_and_move_original_files(bids_out: Path, subj_id: str, ses_id: str):
    tmp_folder = bids_out / "tmp_dcm2bids" / f"sub-{subj_id}_ses-{ses_id}"
    if not tmp_folder.exists():
        return

    sub_dir = bids_out / f"sub-{subj_id}"
    ses_dir = sub_dir / f"ses-{ses_id}" if ses_id else sub_dir
    ses_dir.mkdir(parents=True, exist_ok=True)

    modality_paths = {
        "anat": ses_dir / "anat",
        "dwi":  ses_dir / "dwi",
        "func": ses_dir / "func",
        "perf": ses_dir / "perf"
    }

    # Loop over JSON sidecars only
    for json_file in tmp_folder.rglob("*.json"):
        try:
            with open(json_file, "r") as jf:
                meta = json.load(jf)
        except Exception:
            st.warning(f"Could not read JSON: {json_file.name}")
            continue

        # Check for ORIGINAL in ImageType
        image_type = meta.get("ImageType", [])
        if isinstance(image_type, str):
            image_type = [image_type]
        if not any("original" in item.lower() for item in image_type):
            st.info(f"Discarded non-original: {json_file.name}")
            continue

        # Determine modality from metadata
        desc = (meta.get("SeriesDescription", "") + " " +
                meta.get("ProtocolName", "")).lower()
        pulse = meta.get("PulseSequenceName", "").lower()

        if "t1" in desc and "flair" not in desc:
            modality, suffix = "anat", "T1w"
        elif "t2" in desc:
            modality, suffix = "anat", "T2w"
        elif "flair" in desc or "fluid" in desc:
            modality, suffix = "anat", "FLAIR"
        elif "dwi" in desc or "dti" in desc:
            modality, suffix = "dwi", "dwi"
        elif "bold" in desc or "fmri" in desc or "functional" in desc or "activation" in desc or "epi" in pulse:
            modality, suffix = "func", "bold"
        elif "asl" in desc or "perfusion" in desc:
            modality, suffix = "perf", "asl"
        else:
            st.info(f"Unclassified: {json_file.name}")
            continue

        # Locate matching NIfTI image
        nii_file = json_file.with_suffix(".nii.gz")
        if not nii_file.exists():
            nii_file = json_file.with_suffix(".nii")
        if not nii_file.exists():
            st.warning(f"No matching NIfTI for: {json_file.name}")
            continue

        target_dir = modality_paths[modality]
        target_dir.mkdir(parents=True, exist_ok=True)

        # Compose filenames
        base_name = f"sub-{subj_id}"
        if ses_id:
            base_name += f"_ses-{ses_id}"
        base_name += f"_{suffix}"

        new_json_path = target_dir / f"{base_name}.json"
        new_nii_path = target_dir / (f"{base_name}.nii.gz")

        # Move both
        shutil.move(str(json_file), str(new_json_path))
        shutil.move(str(nii_file), str(new_nii_path))
        st.success(f"Moved: {new_json_path.name} and {new_nii_path.name}")

    # Cleanup
    shutil.rmtree(tmp_folder.parent, ignore_errors=True)
    st.info("Finished organizing ORIGINAL NIfTI + JSON pairs.")


# This line replaces your old move_files_in_tmp()
move_files_in_tmp = classify_and_move_original_files


def create_bids_top_level_files(bids_dir: Path, subject_id: str):
    dd_file = bids_dir / "dataset_description.json"
    if not dd_file.exists():
        dataset_description = {
            "Name": "Example dataset",
            "BIDSVersion": "1.6.0",
            "License": "CC0",
            "Authors": ["Philip Nkwam", "Udunna Anazodo", "Maruf Adewole", "Sekinat Aderibigbe"],
            "DatasetType": "raw"
        }
        with open(dd_file, 'w') as f:
            json.dump(dataset_description, f, indent=4)
    readme_file = bids_dir / "README"
    if not readme_file.exists():
        content = f"""\
# BIDS Dataset

This dataset was automatically generated by dcm2bids.

**Contents**:
- Anat: T1w, T2w, FLAIR
- DWI: Diffusion Weighted Imaging
- Func: BOLD/fMRI scans
- Perf: ASL perfusion scans

Please see the official [BIDS documentation](https://bids.neuroimaging.io) for details.
"""
        with open(readme_file, 'w') as f:
            f.write(content)
    changes_file = bids_dir / "CHANGES"
    if not changes_file.exists():
        content = f"1.0.0 {datetime.datetime.now().strftime('%Y-%m-%d')}\n  - Initial BIDS conversion\n"
        with open(changes_file, 'w') as f:
            f.write(content)
    participants_tsv = bids_dir / "participants.tsv"
    if not participants_tsv.exists():
        with open(participants_tsv, 'w') as f:
            f.write("participant_id\tage\tsex\n")
            f.write(f"sub-{subject_id}\tN/A\tN/A\n")
    participants_json = bids_dir / "participants.json"
    if not participants_json.exists():
        pjson = {
            "participant_id": {"Description": "Unique ID"},
            "age": {"Description": "Age in years"},
            "sex": {"Description": "Biological sex"}
        }
        with open(participants_json, 'w') as f:
            json.dump(pjson, f, indent=4)


def zip_directory(folder_path: Path, zip_file_path: Path):
    shutil.make_archive(str(zip_file_path.with_suffix("")),
                        'zip', root_dir=folder_path)


def extract_iqms_from_html(html_file: Path):
    iqms = {}
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    iqm_table = soup.find("table", {"id": "iqms-table"})
    if iqm_table:
        rows = iqm_table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2:
                metric_name = cols[0].get_text(strip=True)
                metric_value = cols[1].get_text(strip=True)
                iqms[metric_name] = metric_value

    return iqms


def extract_all_iqms(result_dir: Path):
    iqm_list = []
    html_reports = list(result_dir.rglob("*.html"))
    for html_file in html_reports:
        iqms = extract_iqms_from_html(html_file)
        iqms["Report Filename"] = html_file.name
        iqm_list.append(iqms)
    return pd.DataFrame(iqm_list)

# ------------------------------
# Main Streamlit App
# ------------------------------


def main():
    st.title("DICOM → BIDS → MRIQC")

    subj_id = st.text_input("Subject ID (e.g. '01')", value="01")
    ses_id = st.text_input("Session ID (optional)", value="Baseline")

    selected_modalities = st.multiselect(
        "Select MRIQC modalities:",
        ["T1w", "T2w", "bold"],
        default=["T1w"]
    )

    # Resource allocation settings
    col1, col2 = st.columns(2)
    with col1:
        n_procs = st.selectbox(
            "CPU Cores to Use",
            options=[4, 8, 12, 16],
            index=3,  # Default to 16
            help="More cores = faster processing but higher resource usage"
        )
    with col2:
        mem_gb = st.selectbox(
            "Memory Allocation (GB)",
            options=[16, 32, 48, 64],
            index=3,  # Default to 64
            help="More memory allows processing larger datasets"
        )

    API_BASE = "http://52.91.185.103:8000"
    ws_url = "ws://52.91.185.103:8000/ws/mriqc"

    dicom_zip = st.file_uploader("Upload DICOM ZIP", type=["zip"])

    if dicom_zip:
        # Phase 1: DICOM to BIDS Conversion
        if st.button("Run DICOM → BIDS Conversion"):
            with st.spinner("Converting DICOM to BIDS..."):
                job_id = str(uuid.uuid4())[:8]
                temp_dir = Path(f"temp_{job_id}")
                temp_dir.mkdir(exist_ok=True)

                dicom_dir = temp_dir / "dicoms"
                dicom_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(dicom_zip, 'r') as zf:
                    zf.extractall(dicom_dir)
                st.success(f"DICOMs extracted to {dicom_dir}")

                bids_out = temp_dir / "bids_output"
                bids_out.mkdir(exist_ok=True)

                config_file = generate_dcm2bids_config(temp_dir)
                run_dcm2bids(dicom_dir, bids_out, subj_id, ses_id, config_file)

                move_files_in_tmp(bids_out, subj_id, ses_id)
                create_bids_top_level_files(bids_out, subj_id)

                ds_file = bids_out / "dataset_description.json"
                if ds_file.exists():
                    st.success(
                        "dataset_description.json created successfully.")
                else:
                    st.error("dataset_description.json not found at BIDS root!")

                bids_zip_path = temp_dir / "bids_dataset.zip"
                zip_directory(bids_out, bids_zip_path)
                st.success("DICOM to BIDS conversion complete!")
                st.info(f"BIDS dataset is ready: {bids_zip_path}")

                with open(bids_zip_path, "rb") as f:
                    st.download_button(
                        "Download BIDS Dataset",
                        data=f,
                        file_name="BIDS_dataset.zip",
                        mime="application/zip"
                    )

                st.session_state.temp_dir = str(temp_dir)
                st.session_state.bids_zip_path = str(bids_zip_path)
        # Show selected modalities before job submission
        if selected_modalities:
            st.markdown("### ✅ Selected Modalities for MRIQC")
            st.success(
                f"You have selected: `{', '.join(selected_modalities)}`")
        else:
            st.warning("⚠️ No modalities selected!")
        # Phase 2: Send BIDS to AWS for MRIQC Processing
        if st.button("Send BIDS to Web for MRIQC"):
            if "temp_dir" not in st.session_state:
                st.error("No BIDS dataset found. Please run the conversion first.")
                st.stop()

            temp_dir = Path(st.session_state.temp_dir)
            bids_zip_path = st.session_state.bids_zip_path

            # ✅ Ensure current selections are reflected
            modalities_str = " ".join(selected_modalities)
            # 🧪 Log for debugging
            st.info(f"Modalities Sent: {modalities_str}")

            with open(bids_zip_path, 'rb') as f:
                files = {'bids_zip': ('bids_dataset.zip',
                                      f, 'application/zip')}
                metadata = {
                    'participant_label': subj_id,
                    'modalities': modalities_str,
                    'session_id': ses_id or "baseline",
                    'n_procs': str(n_procs),
                    'mem_gb': str(mem_gb)
                }

                st.info(f"Sending ZIP to: {API_BASE}/submit-job")
                t0 = time.time()
                with st.spinner("Submitting job to backend..."):
                    submit_response = requests.post(
                        f"{API_BASE}/submit-job", files=files, data=metadata
                    )
                t1 = time.time()
                st.info(f"Request sent in {t1 - t0:.2f} seconds")

                if submit_response.status_code != 200:
                    st.error(
                        f"MRIQC failed (Status {submit_response.status_code}): {submit_response.text}")
                    st.stop()

                job_id = submit_response.json().get("job_id")
                st.info(f"✅ Job submitted successfully. Job ID: `{job_id}`")

            # Start polling
            with st.spinner("Processing MRIQC... please wait."):
                for attempt in range(120):  # max wait ~20 min
                    time.sleep(10)
                    status_response = requests.get(
                        f"{API_BASE}/job-status/{job_id}")

                    if status_response.status_code != 200:
                        st.warning(
                            f"Polling error (try {attempt + 1}): {status_response.text}")
                        continue

                    result = status_response.json()
                    if result["status"] == "complete":
                        st.success("✅ MRIQC job complete!")

                        # Download result
                        download_url = f"{API_BASE}/download/{job_id}"
                        response = requests.get(download_url)

                        if response.status_code != 200:
                            st.error(
                                f"Failed to download result: {response.text}")
                            break

                        zip_bytes = BytesIO(response.content)

                        # Extract
                        result_dir = temp_dir / "mriqc_results"
                        result_dir.mkdir(exist_ok=True)
                        with zipfile.ZipFile(zip_bytes) as zf:
                            zf.extractall(result_dir)

                        # Parse IQMs
                        iqm_records = []
                        html_reports = list(result_dir.rglob("*.html"))
                        if html_reports:
                            for html_file in html_reports:
                                iqms = extract_iqms_from_html(html_file)
                                iqms["Report Filename"] = html_file.name
                                iqm_records.append(iqms)

                            iqms_df = pd.DataFrame(iqm_records)
                            iqm_csv_path = result_dir / "MRIQC_IQMs.csv"
                            iqms_df.to_csv(iqm_csv_path, index=False)

                            # Rezip
                            updated_zip_path = temp_dir / "mriqc_results_with_IQMs"
                            shutil.make_archive(
                                str(updated_zip_path), 'zip', root_dir=result_dir)

                            # Download button
                            with open(f"{updated_zip_path}.zip", "rb") as f:
                                st.download_button(
                                    label="📥 Download MRIQC Results (including IQMs CSV)",
                                    data=f,
                                    file_name="mriqc_results_with_IQMs.zip",
                                    mime="application/zip"
                                )

                            # View IQMs
                            st.subheader(
                                "Extracted Image Quality Metrics (IQMs)")
                            st.dataframe(iqms_df)

                            # View HTMLs
                            for report in html_reports:
                                with open(report, "r") as rf:
                                    html_data = rf.read()
                                st.components.v1.html(
                                    html_data, height=1000, scrolling=True)
                        else:
                            st.warning(
                                "No HTML reports found in MRIQC results.")

                        # Log
                        log_files = list(result_dir.rglob("mriqc_log.txt"))
                        if log_files:
                            with open(log_files[0], "r") as lf:
                                log_data = lf.read()
                            st.subheader("MRIQC Log")
                            st.text_area("Log Output", log_data, height=400)

                        # 🔥 Clean up
                        cleanup_url = f"{API_BASE}/delete-job/{job_id}"
                        cleanup_response = requests.delete(cleanup_url)
                        if cleanup_response.status_code == 200:
                            st.info("✅ Backend job data cleaned up.")
                        else:
                            st.warning(
                                "⚠️ Failed to clean up job data on server.")
                        break

                    elif result["status"] == "failed":
                        st.error(
                            f"❌ Job failed: {result.get('error', 'Unknown error')}")
                        break

                else:
                    st.warning(
                        "⚠️ MRIQC processing timed out after 20 minutes.")


# ------------------------------
# Footer and Branding
# ------------------------------


# Container with collective padding
st.markdown("""
    <div style="padding: 100px;">
""", unsafe_allow_html=True)

# Adjust column widths to center contents
col1, col2 = st.columns([1, 3])

with col1:
    st.image("MLAB.png", width=250)

with col2:
    st.markdown(
        "<h2 style='padding-top: 40px;'>Medical Artificial Intelligence Lab</h2>",
        unsafe_allow_html=True
    )

# Close container div
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Hide Streamlit's default footer */
    footer { visibility: hidden; }
    /* Custom footer styling */
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9f9f9;
        text-align: center;
        padding: 10px 0;
        border-top: 1px solid #e0e0e0;
        font-size: 14px;
        color: #333;
    }
    .custom-footer img {
        height: 40px;
        vertical-align: middle;
        margin-right: 10px;
    }
    </style>
    <div class="custom-footer">
        <strong>Medical Artificial Intelligence Lab || Contact Email: info@mailab.io </strong> – © 2025 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
