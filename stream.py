import os
import json
import uuid
import time
import shutil
import logging
import zipfile
import threading
import datetime
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://haske.online",
        "https://haske.online:5000",
        "https://haske.online:8090"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job tracker
PROCESSING_JOBS = {}
job_executor = ThreadPoolExecutor(max_workers=4)

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="MRIQC App", layout="wide")

# ------------------------------
# Sidebar with Collapsible Information
# ------------------------------
with st.sidebar:
    st.image("MLAB.png", width=200)  # Adjust path as needed
    st.markdown("### MRIQC Web App")
    
    # Collapsible section for app information
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        The Web-MRIQC App provides an intuitive web interface for running Quality Control on MRI datasets acquired in DICOM formats. The App offers users the ability to compute Image Quality Metrics (IQMs) for neuroimaging studies.
        
        This web-based solution implements the original MRIQC standalone application in a user-friendly interface accessible from any device, without the need for software installation or access to resource-intensive computers. Thus, simplifying the quality control workflow. For a comprehensive understanding of the IQMs computed by MRIQC, as well as details on the original MRIQC implementation, refer to the official MRIQC documentation: [MRIQC Documentation](https://mriqc.readthedocs.io).
        """)
    
    # Collapsible section for usage instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Enter Subject ID (optional)
        2. Enter the Session ID (optional, e.g, baseline, follow up, etc)
        3. Select your preferred modality for analysis (T1w, T2w, DWI, BOLD fMRI, or ASL)
        4. Upload a zipped file/folder containing T1w, T2w, DWI, BOLD fMRI, or ASL DICOM images
        5. Click DICOM → BIDS Conversion
        6. Once BIDS converted, you will see the notification: DICOM to BIDS conversion complete
        7. Click Send BIDS to Web for MRIQC or if you want the BIDS format, Click Download BIDS Dataset to your device.
        8. Send the converted BIDS images to MRIQC by clicking Send BIDS to Web for MRIQC for generating the IQMs
        9. Depending on your internet connection, this can between 5-10 minutes to get your results for a single participant.
        10. When completed, you can view the report on the web App or download the report of the IQM by clicking the "Download MRIQC results" button including the csv export.
        """)
    
    # Collapsible section for references
    with st.expander("📚 References"):
        st.markdown("""
        1. Boré, A., Guay, S., Bedetti, C., Meisler, S., & GuenTher, N. (2023). Dcm2Bids (Version 3.1.1) [Computer software]. https://doi.org/10.5281/zenodo.8436509
        2. Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods., 2016, 264:47-56.
        3. Esteban O, Birman D, Schaer M, Koyejo OO, Poldrack RA, Gorgolewski KJ (2017) MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites. PLoS ONE 12(9): e0184661. https://doi.org/10.1371/journal.pone.0184661
        """)
    
    # Collapsible section for IQM tables
    with st.expander("📊 IQM Definitions"):
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
    
    st.markdown("---")
    st.markdown("""
    **Medical Artificial Intelligence Lab**  
    Contact: info@mailab.io  
    © 2025 All Rights Reserved
    """)

# ------------------------------
# Main App Content
# ------------------------------
st.title("Haske - MRI Image Quality Assessment")
st.caption("Supported by MAI Lab")

# Check if started from Orthanc
query_params = st.experimental_get_query_params()
patient_id_from_orthanc = query_params.get('patient_id', [None])[0]
processing_started = query_params.get('processing', [None])[0] == 'started'

if patient_id_from_orthanc and processing_started:
    st.success(f"🔄 MRIQC processing started for Patient ID: {patient_id_from_orthanc}")
    st.info("This process was initiated from ORTHANC. Processing is running in the background.")
    
    # Show job status if available
    if st.button("Check Processing Status"):
        st.info("Check your Orthanc interface for processing updates, or monitor the server logs.")

# Regular Streamlit interface for manual uploads
st.divider()
st.subheader("Manual Upload (Alternative)")

# Subject information form
with st.form("subject_info"):
    st.subheader("Subject Information")
    col1, col2 = st.columns(2)
    with col1:
        subj_id = st.text_input("Subject ID", value=patient_id_from_orthanc or "01")
    with col2:
        ses_id = st.text_input("Session ID (optional)")
    st.form_submit_button("Update Subject Info")

# Add a status dashboard for background jobs
if st.checkbox("Show Processing Dashboard"):
    st.subheader("Active Processing Jobs")
    
    if PROCESSING_JOBS:
        for job_id, job in PROCESSING_JOBS.items():
            with st.expander(f"Job {job_id} - Patient {job['patient_id']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Status:** {job['status']}")
                    st.write(f"**Created:** {job['created_at']}")
                with col2:
                    st.write(f"**Progress:** {job.get('progress', 'N/A')}")
                    if job.get('error'):
                        st.error(f"Error: {job['error']}")
    else:
        st.info("No active jobs")

# Run FastAPI server in a separate thread when Streamlit starts
if 'fastapi_started' not in st.session_state:
    import uvicorn
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8502, log_level="info")
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    st.session_state.fastapi_started = True
    
    st.success("🚀 Backend API server started on port 8090")

# ------------------------------
# API Endpoints
# ------------------------------

# Temporary storage for uploaded files
UPLOAD_TRACKER: Dict[str, Dict[str, Any]] = {}

@app.post("/api/upload-dicom")
async def upload_dicom_from_orthanc(
    dicom_zip: UploadFile = File(...),
    patient_id: str = Form(...),
    study_instance_uid: str = Form(default=""),
    source: str = Form(default="orthanc"),
    auto_process: bool = Form(default=True)
):
    """
    Endpoint to receive DICOM ZIP from Orthanc and process it
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Create temp directory
        temp_dir = Path(f"temp_{job_id}")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded ZIP
        zip_path = temp_dir / f"{patient_id}_{int(time.time())}.zip"
        with open(zip_path, "wb") as buffer:
            content = await dicom_zip.read()
            buffer.write(content)
        
        logger.info(f"Received DICOM ZIP for patient {patient_id}, size: {len(content)} bytes")
        
        # Store job info
        PROCESSING_JOBS[job_id] = {
            "status": "received",
            "patient_id": patient_id,
            "study_instance_uid": study_instance_uid,
            "temp_dir": str(temp_dir),
            "zip_path": str(zip_path),
            "created_at": datetime.datetime.now().isoformat(),
            "progress": "DICOM received successfully"
        }
        
        # If auto_process is True, start processing in background
        if auto_process:
            # Submit processing job to thread pool
            future = job_executor.submit(process_dicom_job, job_id)
            PROCESSING_JOBS[job_id]["future"] = future
        
        return {
            "success": True,
            "job_id": job_id,
            "message": f"DICOM received for patient {patient_id}",
            "auto_processing": auto_process
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status for a job"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PROCESSING_JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "patient_id": job["patient_id"],
        "progress": job.get("progress", ""),
        "created_at": job["created_at"],
        "error": job.get("error")
    }

@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    """Download results for completed job"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PROCESSING_JOBS[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")
    
    results_path = job.get("results_path")
    if not results_path or not Path(results_path).exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Return file for download
    return FileResponse(
        results_path,
        media_type='application/zip',
        filename=f"mriqc_results_{job_id}.zip"
    )

@app.post("/api/start-processing/{job_id}")
async def start_processing(job_id: str):
    """Manually start processing for a job"""
    if job_id not in PROCESSING_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PROCESSING_JOBS[job_id]
    if job["status"] != "received":
        raise HTTPException(status_code=400, detail="Job already processing or complete")
    
    # Start processing
    future = job_executor.submit(process_dicom_job, job_id)
    PROCESSING_JOBS[job_id]["future"] = future
    
    return {"success": True, "message": "Processing started"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_jobs": len([j for j in PROCESSING_JOBS.values() if j["status"] == "processing"]),
        "total_jobs": len(PROCESSING_JOBS),
        "timestamp": datetime.datetime.now().isoformat()
    }

# ------------------------------
# Background Processing Functions
# ------------------------------

def process_dicom_job(job_id: str):
    """Process DICOM conversion and MRIQC in background thread"""
    try:
        job = PROCESSING_JOBS[job_id]
        job["status"] = "processing"
        job["progress"] = "Starting DICOM to BIDS conversion..."
        
        temp_dir = Path(job["temp_dir"])
        zip_path = Path(job["zip_path"])
        patient_id = job["patient_id"]
        
        # Extract DICOMs
        job["progress"] = "Extracting DICOM files..."
        dicom_dir = temp_dir / "dicoms"
        dicom_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dicom_dir)
        
        # Convert to BIDS
        job["progress"] = "Converting to BIDS format..."
        bids_out = temp_dir / "bids_output"
        bids_out.mkdir(exist_ok=True)
        
        # Use session as empty string to avoid session folders
        ses_id = ""
        
        config_file = generate_dcm2bids_config(temp_dir)
        run_dcm2bids(dicom_dir, bids_out, patient_id, ses_id, config_file)
        classify_and_move_original_files(bids_out, patient_id, ses_id)
        create_bids_top_level_files(bids_out, patient_id)
        
        # Create BIDS ZIP
        bids_zip_path = temp_dir / "bids_dataset.zip"
        zip_directory(bids_out, bids_zip_path)
        
        job["progress"] = "BIDS conversion complete. Starting MRIQC..."
        job["bids_zip_path"] = str(bids_zip_path)
        
        # Run MRIQC
        modalities = ["T1w"]  # Default, can be made configurable
        results_path = run_mriqc_processing(bids_zip_path, patient_id, modalities, job_id)
        
        # Mark as complete
        job["status"] = "complete"
        job["progress"] = "MRIQC processing complete!"
        job["results_path"] = results_path
        job["completed_at"] = datetime.datetime.now().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        PROCESSING_JOBS[job_id]["status"] = "failed"
        PROCESSING_JOBS[job_id]["error"] = str(e)
        PROCESSING_JOBS[job_id]["progress"] = f"Failed: {str(e)}"

def run_mriqc_processing(bids_zip_path: Path, patient_id: str, modalities: list, job_id: str):
    """Run MRIQC processing via API"""
    try:
        job = PROCESSING_JOBS[job_id]
        
        # Your existing MRIQC API processing code
        API_BASE = "http://52.91.185.103:8000"
        
        with open(bids_zip_path, 'rb') as f:
            files = {'bids_zip': ('bids_dataset.zip', f, 'application/zip')}
            metadata = {
                'participant_label': patient_id,
                'modalities': " ".join(modalities),
                'session_id': "baseline",
                'n_procs': "16",
                'mem_gb': "64"
            }

            job["progress"] = "Submitting to MRIQC service..."
            submit_response = requests.post(
                f"{API_BASE}/submit-job", 
                files=files, 
                data=metadata
            )

        if submit_response.status_code != 200:
            raise Exception(f"MRIQC submission failed: {submit_response.text}")

        mriqc_job_id = submit_response.json().get("job_id")
        job["progress"] = f"MRIQC job submitted (ID: {mriqc_job_id})"
        
        # Poll for results
        result = None
        for attempt in range(120):  # 20 minute timeout
            time.sleep(10)
            status_response = requests.get(f"{API_BASE}/job-status/{mriqc_job_id}")

            if status_response.status_code == 200:
                result = status_response.json()
                if result["status"] == "complete":
                    break
                elif result["status"] == "failed":
                    raise Exception(f"MRIQC processing failed: {result.get('error')}")
                    
            job["progress"] = f"MRIQC processing... (attempt {attempt + 1}/120)"

        if not result or result["status"] != "complete":
            raise Exception("MRIQC processing timed out")

        # Download results
        job["progress"] = "Downloading MRIQC results..."
        download_url = f"{API_BASE}/download/{mriqc_job_id}"
        response = requests.get(download_url)
        
        if response.status_code != 200:
            raise Exception(f"Results download failed: {response.text}")

        # Save results
        temp_dir = Path(job["temp_dir"])
        results_zip_path = temp_dir / f"mriqc_results_{job_id}.zip"
        with open(results_zip_path, 'wb') as f:
            f.write(response.content)

        # Clean up MRIQC job
        requests.delete(f"{API_BASE}/delete-job/{mriqc_job_id}")
        
        return str(results_zip_path)
        
    except Exception as e:
        logger.error(f"MRIQC processing failed for job {job_id}: {str(e)}")
        raise

# ------------------------------
# DICOM to BIDS Conversion Functions
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

# ------------------------------
# MRIQC Report Processing
# ------------------------------

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
# Streamlit Processing Interface
# ------------------------------

def main():
    # Processing options section
    st.divider()
    st.subheader("Processing Options")
    
    # Modality selection
    selected_modalities = st.multiselect(
        "Select MRIQC modalities:",
        ["T1w", "T2w", "bold"],
        default=["T1w"],
        help="Select which MRI modalities to analyze"
    )
    
    # Resource allocation settings
    with st.expander("Advanced Resource Settings"):
        col1, col2 = st.columns(2)
        with col1:
            n_procs = st.selectbox(
                "CPU Cores to Use",
                options=[4, 8, 12, 16],
                index=3,
                help="More cores = faster processing but higher resource usage"
            )
        with col2:
            mem_gb = st.selectbox(
                "Memory Allocation (GB)",
                options=[16, 32, 48, 64],
                index=3,
                help="More memory allows processing larger datasets"
            )
    
    # DICOM upload section
    st.divider()
    st.subheader("Data Upload")
    dicom_zip = st.file_uploader("Upload DICOM ZIP", type=["zip"])
    
    # Only show processing options when DICOM is ready
    if st.session_state.get("dicom_ready"):
        if st.button("Start MRIQC Analysis", type="primary"):
            with st.spinner("Processing..."):
                file_path = UPLOAD_TRACKER.get(st.session_state.upload_id, {}).get("file_path")
                if file_path:
                    st.success(f"Processing file: {file_path}")
                else:
                    st.error("No file found for the current upload ID")
    
    # DICOM to BIDS conversion
    if dicom_zip:
        st.divider()
        st.subheader("DICOM Conversion")
        
        if st.button("Run DICOM → BIDS Conversion", type="primary"):
            with st.spinner("Converting DICOM to BIDS..."):
                try:
                    job_id = str(uuid.uuid4())[:8]
                    temp_dir = Path(f"temp_{job_id}")
                    temp_dir.mkdir(exist_ok=True)

                    # Extract DICOMs
                    dicom_dir = temp_dir / "dicoms"
                    dicom_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(dicom_zip, 'r') as zf:
                        zf.extractall(dicom_dir)
                    st.success(f"DICOMs extracted to {dicom_dir}")

                    # Convert to BIDS
                    bids_out = temp_dir / "bids_output"
                    bids_out.mkdir(exist_ok=True)
                    config_file = generate_dcm2bids_config(temp_dir)
                    run_dcm2bids(dicom_dir, bids_out, subj_id, ses_id, config_file)
                    move_files_in_tmp(bids_out, subj_id, ses_id)
                    create_bids_top_level_files(bids_out, subj_id)

                    # Verify conversion
                    ds_file = bids_out / "dataset_description.json"
                    if ds_file.exists():
                        st.success("BIDS conversion successful!")
                    else:
                        st.error("BIDS conversion failed - missing dataset_description.json")

                    # Create downloadable zip
                    bids_zip_path = temp_dir / "bids_dataset.zip"
                    zip_directory(bids_out, bids_zip_path)
                    
                    # Store in session state
                    st.session_state.temp_dir = str(temp_dir)
                    st.session_state.bids_zip_path = str(bids_zip_path)
                    
                    # Download button
                    with open(bids_zip_path, "rb") as f:
                        st.download_button(
                            "Download BIDS Dataset",
                            data=f,
                            file_name="BIDS_dataset.zip",
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Error during conversion: {str(e)}")
                    st.exception(e)

    # MRIQC Processing Section
    if "temp_dir" in st.session_state and selected_modalities:
        st.divider()
        st.subheader("MRIQC Processing")
        
        if st.button("Send BIDS to Web for MRIQC", type="primary"):
            try:
                temp_dir = Path(st.session_state.temp_dir)
                bids_zip_path = st.session_state.bids_zip_path
                modalities_str = " ".join(selected_modalities)

                with st.status("Submitting MRIQC job...", expanded=True) as status:
                    # Upload BIDS data
                    with open(bids_zip_path, 'rb') as f:
                        files = {'bids_zip': ('bids_dataset.zip', f, 'application/zip')}
                        metadata = {
                            'participant_label': subj_id,
                            'modalities': modalities_str,
                            'session_id': ses_id or "baseline",
                            'n_procs': str(n_procs),
                            'mem_gb': str(mem_gb)
                        }

                        status.write("Uploading BIDS data...")
                        submit_response = requests.post(
                            f"{API_BASE}/submit-job", 
                            files=files, 
                            data=metadata
                        )

                    if submit_response.status_code != 200:
                        status.error(f"Submission failed: {submit_response.text}")
                        return

                    job_id = submit_response.json().get("job_id")
                    status.success(f"Job submitted (ID: {job_id})")
                    
                    # Poll for results
                    status.write("Processing MRIQC...")
                    result = None
                    for attempt in range(120):  # ~20 minute timeout
                        time.sleep(10)
                        status_response = requests.get(f"{API_BASE}/job-status/{job_id}")

                        if status_response.status_code != 200:
                            status.warning(f"Polling error (attempt {attempt + 1})")
                            continue

                        result = status_response.json()
                        if result["status"] == "complete":
                            status.success("Processing complete!")
                            break
                        elif result["status"] == "failed":
                            status.error(f"Processing failed: {result.get('error')}")
                            return

                    if not result or result["status"] != "complete":
                        status.warning("Processing timed out")
                        return

                    # Download and display results
                    status.write("Downloading results...")
                    download_url = f"{API_BASE}/download/{job_id}"
                    response = requests.get(download_url)
                    
                    if response.status_code != 200:
                        status.error(f"Download failed: {response.text}")
                        return

                    # Process results
                    zip_bytes = BytesIO(response.content)
                    result_dir = temp_dir / "mriqc_results"
                    result_dir.mkdir(exist_ok=True)
                    
                    with zipfile.ZipFile(zip_bytes) as zf:
                        zf.extractall(result_dir)

                    # Extract IQMs from HTML reports
                    html_reports = list(result_dir.rglob("*.html"))
                    if html_reports:
                        iqm_records = []
                        for html_file in html_reports:
                            iqms = extract_iqms_from_html(html_file)
                            iqms["Report Filename"] = html_file.name
                            iqm_records.append(iqms)

                        # Create and display IQMs table
                        iqms_df = pd.DataFrame(iqm_records)
                        st.dataframe(iqms_df)

                        # Create downloadable package
                        iqm_csv_path = result_dir / "MRIQC_IQMs.csv"
                        iqms_df.to_csv(iqm_csv_path, index=False)
                        updated_zip_path = temp_dir / "mriqc_results_with_IQMs"
                        shutil.make_archive(str(updated_zip_path), 'zip', root_dir=result_dir)

                        # Download button
                        with open(f"{updated_zip_path}.zip", "rb") as f:
                            st.download_button(
                                label="Download Full Results",
                                data=f,
                                file_name="mriqc_results.zip",
                                mime="application/zip"
                            )

                        # Display HTML reports
                        for report in html_reports:
                            with st.expander(f"View Report: {report.name}"):
                                with open(report, "r") as rf:
                                    st.components.v1.html(rf.read(), height=1000, scrolling=True)
                    
                    # Clean up
                    requests.delete(f"{API_BASE}/delete-job/{job_id}")

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
