CUDA_VISIBLE_DEVICES=0 python test_graspnet.py \
    --center-num 48 \
    --anchor-num 7 \
    --anchor-k 6 \
    --anchor-w 50 \
    --anchor-z 20 \
    --grid-size 8 \
    --all-points-num 25600 \
    --group-num 512 \
    --local-k 10 \
    --ratio 8 \
    --input-h 360 \
    --input-w 640 \
    --local-thres 0.01 \
    --heatmap-thres 0.01 \
    --num-workers 4 \
    --dataset-path '/data/hggd_dataset_v2/grasp_label' \
    --scene-path '/data/hggd_dataset_v2' \
    --checkpoint-path '/home/hggd/logs/260617_092444_proberoehrchen_v2/epoch_14_iou_1.000_cover_0.015' \
    --frames-per-scene 50 \
    --description 'proberoehrchen_test_1' \
    --vis-frame-ids "1,17,124,189,344,469" \

#--checkpoint-path '/home/hggd/HGGD_realsense_checkpoint' \
#--checkpoint-path '/home/hggd/logs/260526_141810_proberoehrchen_v2/epoch_14_iou_1.000_cover_0.006' \

