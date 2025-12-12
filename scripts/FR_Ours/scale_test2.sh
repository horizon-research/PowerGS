export CUDA_VISIBLE_DEVICES=1
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1224 5
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1224 6
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1224 7
bash compose_a_scene.sh bicycle FR_Ours_0.99_5
bash compose_a_scene.sh bicycle FR_Ours_0.99_6
bash compose_a_scene.sh bicycle FR_Ours_0.99_7
bash eval_a_scene.sh bicycle FR_Ours_0.99_5
bash eval_a_scene.sh bicycle FR_Ours_0.99_6
bash eval_a_scene.sh bicycle FR_Ours_0.99_7
