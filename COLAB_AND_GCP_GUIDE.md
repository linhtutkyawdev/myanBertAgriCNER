# Guide: Running Fine-tuning on Google Colab and Google Cloud (GCP)

Training or fine-tuning models like `google/mt5-small` on Burmese NER can be extremely slow on CPU. Leveraging a GPU via **Google Colab** (free/pro managed environment) or **Google Cloud Platform (GCP)** via `gcloud` will speed up training from hours/days to just a few minutes.

Below are complete, step-by-step instructions for both environments.

---

## Method 1: Google Colab (Recommended for Simplicity)

Google Colab provides a free or low-cost T4 GPU, which is perfect for this task. You can run training interactively.

### Steps to Run on Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Change the runtime type to GPU:
   - Go to **Runtime** > **Change runtime type** > Select **T4 GPU** (or any available GPU) > Click **Save**.
4. Paste and execute the following cells in your notebook:

#### Cell 1: Clone the Repository and Navigate In
```bash
# Clone the repository (replace with your repo URL)
!git clone https://github.com/linhtutkyawdev/myanBertAgriCNER.git
%cd myanBertAgriCNER
```

#### Cell 2: Install Dependencies using `uv` or `pip`
Since the project uses `pyproject.toml`, we can install dependencies very quickly using `pip` (or `uv` if you prefer, but standard `pip` works out of the box in Colab):
```bash
# Install the package and all its dependencies
!pip install .
```

#### Cell 3: Prepare the Dataset
This script parses the raw text file, generates labels, splits the dataset into train/validation/test sets deterministically, and saves them.
```python
!python scripts/prepare_data.py
```

#### Cell 4: Run Fine-Tuning on GPU
The training script automatically detects the GPU and configures high-performance settings (FP16, batch size 16, etc.):
```python
!python scripts/train_mt5.py
```

#### Cell 5: Test Model Predictions
```python
!python scripts/predict.py --model mt5 --text "စပါး စိုက်ပျိုး ရာတွင် ဂျစ်ဆန် နှင့် ယူရီးယား ကို ၂ကြိမ် ခွဲ၍သုံးပါ၊၊"
```

#### Cell 6: Persist Model Checkpoints to Google Drive (Optional)
To save your trained model so it doesn't get wiped when the Colab instance shuts down:
```python
from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Copy the trained model weights to Google Drive
shutil.copytree("experiments/mt5/best_model", "/content/drive/MyDrive/myanbert_cner_mt5_best_model")
print("Model copied to Google Drive successfully!")
```

---

## Method 2: Google Cloud Platform (GCP) via `gcloud`

If you want a dedicated machine or need to automate the pipeline, Google Cloud Platform (GCP) is the best choice. You have two main approaches: **Compute Engine (VMs)** and **Vertex AI (Managed Custom Jobs)**.

---

### Option A: GPU-Enabled Compute Engine VM (Direct VM Access)

Creating a Google Compute Engine instance with a pre-configured Deep Learning Image (with CUDA, PyTorch, and Python pre-installed) is the simplest way to run on GCP.

#### Step 1: Create a GPU Instance using `gcloud`
Run this command in your local terminal (make sure you have initialized the Cloud SDK with `gcloud init` and have GPU quota):

```bash
gcloud compute instances create myanberta-cner-gpu-vm \
    --project="YOUR_PROJECT_ID" \
    --zone="us-central1-a" \
    --machine-type="n1-standard-4" \
    --maintenance-policy="TERMINATE" \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="100GB" \
    --boot-disk-type="pd-ssd" \
    --metadata="install-nvidia-driver=true"
```

#### Step 2: SSH into the Instance
```bash
gcloud compute ssh myanberta-cner-gpu-vm --zone="us-central1-a"
```

#### Step 3: Run the Training inside the VM
Once inside the VM, execute:
```bash
# Clone and enter the project
git clone https://github.com/linhtutkyawdev/myanBertAgriCNER.git
cd myanBertAgriCNER

# Install dependencies
pip install .

# Prepare data and start training
python scripts/prepare_data.py
python scripts/train_mt5.py
```

#### Step 4: Shut Down / Delete Instance to Save Cost
**CRITICAL:** GPUs are expensive. Always stop or delete the instance once training completes:
```bash
# To stop (can be resumed later)
gcloud compute instances stop myanberta-cner-gpu-vm --zone="us-central1-a"

# To delete permanently
gcloud compute instances delete myanberta-cner-gpu-vm --zone="us-central1-a"
```

---

### Option B: Vertex AI Custom Training Job (Serverless / Managed)

Vertex AI Custom Training lets you submit a script to run on a managed GPU node. The node is provisioned, runs your code, saves the results to a Cloud Storage (GCS) bucket, and shuts down automatically.

#### Step 1: Prepare a Cloud Storage Bucket
```bash
gsutil mb gs://myanberta-cner-bucket-unique-name/
```

#### Step 2: Create a `config.yaml` for your Custom Job
Create a configuration file to tell Vertex AI what machine type and GPU accelerator to use.

```yaml
# config.yaml
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    # Google prebuilt PyTorch GPU image
    imageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0.py310:latest
    args: [
      "git clone https://github.com/linhtutkyawdev/myanBertAgriCNER.git && \
       cd myanBertAgriCNER && \
       pip install . && \
       python scripts/prepare_data.py && \
       python scripts/train_mt5.py && \
       gsutil -m cp -r experiments/mt5/best_model gs://myanberta-cner-bucket-unique-name/best_model"
    ]
    command: ["/bin/bash", "-c"]
```

#### Step 3: Submit the Training Job using `gcloud`
Submit the custom training job to Vertex AI:

```bash
gcloud ai custom-jobs create \
    --region="us-central1" \
    --display-name="myanbert-cner-mt5-training" \
    --config="config.yaml"
```

Vertex AI will:
1. Provision an `n1-standard-4` VM with an `NVIDIA T4` GPU.
2. Spin up the official PyTorch container.
3. Clone your code, run the data preparation, and execute fine-tuning.
4. Copy the trained `best_model` folder back to your Cloud Storage bucket.
5. Tear down the VM automatically, ensuring you never pay for idle GPU time.

---

## Summary Recommendation

* **For active development and debugging:** Use **Google Colab**. It's interactive, free, and lets you inspect output immediately.
* **For headless automated runs / production lines:** Use **GCP Vertex AI Custom Jobs** because it handles provisioning and deprovisioning automatically, saving you from accidental bills.
* **For complete environment control:** Use **Compute Engine Deep Learning VMs**, but remember to stop them!
