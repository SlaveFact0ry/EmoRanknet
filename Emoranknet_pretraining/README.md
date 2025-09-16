Pre-training the X3D Intensity Feature Extractor with RankNet
This document outlines the process for pre-training your X3D model using a pairwise ranking loss (RankNet). The goal is to fine-tune the model so that it outputs a feature embedding whose magnitude is directly correlated with the perceived emotional intensity in a video.

1. Objective
   Instead of just classifying an emotion, we want to teach the model the concept of relative intensity. Given two videos, the model should be able to predict which one is more emotionally intense.

By learning this, the model's internal feature representations (the embeddings from the penultimate layer) will become highly ordered and meaningful for our downstream TTS task. A more intense visual expression will result in a feature vector with a larger magnitude, providing a rich, continuous signal for controlling vocal prosody.

2. Methodology: Learning to Rank with RankNet
   We frame this as a "learning to rank" problem.

Data Pairing: We don't train on single videos. Instead, we create pairs (Video A, Video B).

Target Label (t): Based on your ground-truth intensity scores from the annotation phase, we create a target label for each pair:

If Intensity(A) > Intensity(B), the target t is 1.

If Intensity(B) > Intensity(A), the target t is 0.

If their intensities are equal, the target t is 0.5.

Model Scoring: The model processes both videos and outputs a scalar "relevance score" for each (s_A, s_B). This score represents the model's prediction of the video's intensity.

RankNet Loss: The loss function penalizes the model when its predicted ranking is incorrect. Specifically, it uses the logistic loss on the difference of the scores (s_A - s_B) to learn the correct relative ordering.

3. How to Use the pretrain_x3d_ranknet.py Script
   Step 1: Prepare Your Data
   You need a single CSV file that maps your video files to their numerical intensity scores. This is the output of your annotation phase. The script expects this CSV to have at least two columns:

video_path: The relative path to the video file.

intensity_score: The numerical intensity score for that video.

Step 2: Customize the Script
Open pretrain_x3d_ranknet.py and modify the following placeholder sections:

load_your_x3d_model(): Replace the placeholder with the code to load your specific pre-trained X3D model.

preprocess_video_for_x3d(): This is the most important part. Replace the placeholder with your actual video loading and preprocessing pipeline (e.g., frame sampling, resizing, normalization).

Model Architecture: Inside the IntensityScorer class, double-check that the penultimate_layer reference correctly points to the layer in your X3D model from which you want to extract features.

Step 3: Run the Training
Execute the script from your terminal. Make sure to provide the paths to your data and an output directory for saving checkpoints.

python pretrain_x3d_ranknet.py \
 --data_csv "/path/to/your/intensity_annotations.csv" \
 --video_root "/path/to/your/video_files_directory/" \
 --output_dir "./x3d_ranknet_training_output" \
 --epochs 20 \
 --batch_size 16 \
 --lr 1e-5

The script will automatically split your data into training and validation sets, train the model, and save the model with the best validation accuracy to the specified output directory.

4. Output and Next Steps
   The script will save the best-performing model weights (e.g., best_model.pth). This saved file contains the weights for your entire IntensityScorer model.

For the next stage (the TTS project), you will need to:

Instantiate the IntensityScorer model again.

Load these pre-trained weights into it.

Use the feature extractor part (IntensityScorer.feature_extractor) of this loaded model inside your extract_x3d_features.py script to generate the final .npy embeddings for the TTS training.
