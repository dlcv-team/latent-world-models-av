# V-JEPA2 Multi-Frame Ablation (A19)

V-JEPA2 with 64-frame temporal input achieves a steer scene-mean RMSE of
0.0579, which is 44% lower
than the best single-frame encoder (DINOv2-S at 0.1042)
and 41% lower than V-JEPA2 in single-frame mode
(0.0974). At single-frame input, V-JEPA2 rep1
(0.1175 combined RMSE) and DINOv2-S (0.1192)
are within 1.4% of each other, suggesting the
architectural differences between the two encoders matter less than the
availability of temporal context. Notably, the rep1-vs-rep64 gap was absent in
the 240-scene pilot (both ~0.121 combined RMSE) but emerges clearly at 850
scenes, likely because the larger and more diverse training set provides enough
variation for temporal features to express their advantage over single-frame
representations.
