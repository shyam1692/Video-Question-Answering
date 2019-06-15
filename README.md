# Video-Question-Answering
This is a less explored research area, currently work in progress.
So far, literature review has been done on Visual Question Answering and Video Question Answering.

## Methodology at high level:
1. Use Action recognition pipeline to compute features from the video, removing the last fully connected layer of the network.
   The algorithm used is Temporal Segment Networks, which takes two stream approach for action recognition.
2. The feature embeddings for textual data has been done using Glove vectors (300 dimension).
3. Element wise dot product has been taken on the transformed video features and Glove vectors, which we believe would give similarity between specific frames and textual question.
