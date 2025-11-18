Note:

The annex will be used as support materials of this paper "Ultra-Minimal Strain-Based Tactile HCI Device for Force/Position Interaction in Severe Upper-Limb Impairments".
These codes are just for peer review and academic exchanges. Please don't make them for other commercial use without permission. If you are interested in our research work or have any questions, please do not hesitate to contact us.
To comply with the journal’s page limit (less than ten pages) for the initial submission, some context of this paper has been made available at: (https://github.com/HuntingStar/UMSTD_Supplementary_Information). The detailed file directory and description are shown as follows.

├─ codes ## Codes File. These codes are used to train the model proposed in this paper.

├─ CNN ## Model code for training model location coarse classification in UMSTS location prediction

├─ MLP ## This file is used to train the force/position accurate prediction model for
each region after the rough classification of the model in the UMSTD position prediction.

Brief introduction about experimental codes: Although all modules of the experiment have been carefully tested on our own computer, we cannot guarantee that the codes will function equally well on other platforms. It should be noted that some modules depend on specific system configurations and software versions. For instance, the CNN module requires Windows 11, Python 3.12 with PyTorch 2.6.0, and was trained on a single NVIDIA GeForce RTX 4070 GPU (4,608 CUDA cores, 12 GB VRAM) under CUDA 12. In addition, the visual perception and localization module occasionally crashes during execution, and the cause of this issue is not yet fully understood. At present, the codes represent an original version that may contain bugs. We are in the process of thoroughly checking, standardizing, and encapsulating them. A stable and published version will be made available on our GitHub repository in the future.
