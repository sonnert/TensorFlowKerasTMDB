# TensorFlowTMDB

Reads and pre-processes TMDB data, writes protobuffer tfrecords, reads tfrecords and trains simple regressor. Achieves around 0.7 MAE on vote average.

```INFO:tensorflow:Saving dict for global step 910: average_loss = 0.6886058, global_step = 910, label/mean = 6.3343754, loss = 22.035385, prediction/mean = 5.7875543```