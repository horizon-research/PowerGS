export CUDA_VISIBLE_DEVICES=0
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1234 2
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1234 3
# bash true_fit_a_scene_scale_display.sh bicycle 0.99 1234 4
bash compose_a_scene.sh bicycle FR_Ours_0.99_2
bash compose_a_scene.sh bicycle FR_Ours_0.99_3
bash compose_a_scene.sh bicycle FR_Ours_0.99_4    
bash eval_a_scene.sh bicycle FR_Ours_0.99_2
bash eval_a_scene.sh bicycle FR_Ours_0.99_3
bash eval_a_scene.sh bicycle FR_Ours_0.99_4