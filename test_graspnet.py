"""
test_graspnet.py
================

Patched version for VIRAS Probenröhrchen-Datensatz:
  - Liest Test-Szenen aus splits.json (analog zu train_graspnet.py)
  - Nutzt --frames-per-scene Argument
  - cupoch-Import entfernt (Blackwell-Inkompatibilität)
  - GraspNetEval-basierte evaluate() ersetzt durch eigene Funktion,
    die NPZ-Labels mit Predictions vergleicht (analog zu validate()
    in train_graspnet.py).

Aufruf:
    CUDA_VISIBLE_DEVICES=0 python test_graspnet.py \\
        --center-num 48 --anchor-num 7 --anchor-k 6 \\
        --anchor-w 50 --anchor-z 20 --grid-size 8 \\
        --all-points-num 25600 --group-num 512 \\
        --local-k 10 --ratio 8 \\
        --input-h 360 --input-w 640 \\
        --num-workers 4 \\
        --dataset-path '/data/hggd_dataset/grasp_label' \\
        --scene-path '/data/hggd_dataset' \\
        --checkpoint-path '/home/hggd/logs/proberoehrchen_v1/epoch_11.pt' \\
        --frames-per-scene 50 \\
        --dump-dir 'pred_grasps_proberoehrchen' \\
        --description 'proberoehrchen_test'
"""

import argparse
import datetime
import json
import logging
import os
import random
from time import time

# import cupoch  # disabled: not used, Blackwell-incompatible
import numpy as np
import open3d as o3d
import torch
from scipy.signal import medfilt2d
from torch.utils.data import DataLoader

from customgraspnetAPI import Grasp as GraspNetGrasp
from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
from dataset.config import camera
from dataset.evaluation import (anchor_output_process, calculate_6d_match,
                                calculate_coverage, calculate_iou_match,
                                collision_detect, detect_2d_grasp,
                                detect_6d_grasp_multi)
from dataset.grasp import GraspGroup as HGGDGraspGroup
from dataset.grasp import RectGraspGroup
from dataset.graspnet_dataset import GraspnetPointDataset
from dataset.pc_dataset_tools import (data_process, feature_fusion,
                                      get_ori_grasp_label)
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet


# =============================================================================
# Argumente
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', required=True,
                    help='Pfad zum trainierten Checkpoint (.pt)')

# Dataset
parser.add_argument('--dataset-path', required=True,
                    help='Pfad zum grasp_label-Verzeichnis')
parser.add_argument('--scene-path', required=True,
                    help='Pfad zum Eltern-Verzeichnis von scenes/ (mit splits.json)')
parser.add_argument('--scene-l', type=int, default=0,
                    help='Fallback wenn keine splits.json: untere Scene-ID')
parser.add_argument('--scene-r', type=int, default=100,
                    help='Fallback wenn keine splits.json: obere Scene-ID (exclusive)')
parser.add_argument('--frames-per-scene', type=int, default=256,
                    help='Frames pro Szene (GraspNet: 256, VIRAS: 50)')
parser.add_argument('--grasp-count', type=int, default=5000)
parser.add_argument('--dump-dir', default='./pred/test',
                    help='Verzeichnis für die Predictions (NPY pro Frame)')
parser.add_argument('--num-workers', type=int, default=4)

# 2D-Anchor-Net
parser.add_argument('--input-h', type=int, required=True)
parser.add_argument('--input-w', type=int, required=True)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w', type=float, default=50.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=8)

# Punktwolken-Net
parser.add_argument('--anchor-num', type=int, required=True)
parser.add_argument('--all-points-num', type=int, required=True)
parser.add_argument('--center-num', type=int, required=True)
parser.add_argument('--group-num', type=int, required=True)
parser.add_argument('--local-grasp-num', type=int, default=5000)

# Grasp-Detection-Schwellen
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# Evaluation
parser.add_argument('--skip-eval', action='store_true',
                    help='Überspringt die Evaluation, dumpt nur Predictions')
parser.add_argument('--vis-frame-ids', type=str, default='',
                    help='Komma-getrennte Frame-Indizes für Open3D-Visualisierung')

# Logging
parser.add_argument('--logdir', type=str, default='./logs/')
parser.add_argument('--random-seed', type=int, default=123)
parser.add_argument('--description', type=str, default='')


# Translations-/Rotationsschwellen für calculate_6d_match (Eval)
parser.add_argument('--dis-criterion', type=float, default=0.05,
                    help='Translationsschwelle [m] fuer calculate_6d_match')
parser.add_argument('--rot-criterion', type=float, default=0.25,
                    help='Rotationsschwelle [rad] fuer calculate_6d_match')

# Optional: eigene Schwellen für IoU/Coverage
parser.add_argument('--iou-thre', type=float, default=0.25,
                    help='IoU-Schwelle fuer calculate_iou_match')
parser.add_argument('--iou-topk', type=int, default=1,
                    help='Wie viele 2D-Vorschlaege geprueft werden (HGGD: nur Top-1)')
parser.add_argument('--coverage-dis', type=float, default=0.02,
                    help='Translationsschwelle [m] fuer calculate_coverage')
parser.add_argument('--coverage-rot', type=float, default=0.1,
                    help='Rotationsschwelle [rad] fuer calculate_coverage')

args = parser.parse_args()


# =============================================================================
# Konstanten für Eval (gleich wie in train_graspnet.py)
# =============================================================================
dis_criterion = args.dis_criterion
rot_criterion = args.rot_criterion
eval_scale = [0.2, 0.4, 0.6, 0.8, 1.0]


# Werte, die fest in dataset/evaluation.py bzw. dataset/grasp.py stehen.
# Nicht konfigurierbar -- aber sie gehoeren in die Metadaten, sonst ist in
# sechs Monaten nicht mehr nachvollziehbar, unter welchen Bedingungen
# gemessen wurde.
EVAL_CONSTANTS = {
    'collision_voxel_size': 0.01,       # ModelFreeCollisionDetector
    'collision_approach_dist': 0.05,    # mfcdetector.detect
    'collision_mode': 'graspnet',
    'grasp_depth': 0.02,                # to_6d_grasp_group(depth=...)
    'gamma_limit_rad': float(np.pi / 3),  # Filter in detect_6d_grasp_multi
    'rot_uses_soft_distance': True,     # correct_rot prueft mit 2x threshold_dis
    'measured_on': 'rohe 6D-Vorhersagen vor collision_detect und NMS',
}

# ============================================================================
# Hilfsfunktionen
# ============================================================================
def thre_key(prefix, value):
    """Stabiler Schluessel fuer eine Schwelle: 0.010000000000000002 -> 'trans_0.01'."""
    return f'{prefix}_{round(float(value), 6):g}'


# =============================================================================
# Test-Szenen aus splits.json laden
# =============================================================================
def load_test_scenes(scene_path):
    """
    Lädt die Test-Scene-IDs aus splits.json, falls vorhanden,
    sonst Fallback auf --scene-l/--scene-r.
    """
    splits_file = os.path.join(scene_path, 'splits.json')
    if os.path.exists(splits_file):
        with open(splits_file) as f:
            splits = json.load(f)
        test_ids = sorted(splits['test'])
        logging.info(f'Using splits.json test set: {len(test_ids)} scenes')
        logging.info(f'  Scene IDs: {test_ids}')
        return test_ids
    else:
        ids = list(range(args.scene_l, args.scene_r))
        logging.warning(f'No splits.json at {splits_file}, '
                        f'falling back to range({args.scene_l}, {args.scene_r}): '
                        f'{len(ids)} scenes')
        return ids


# =============================================================================
# Inferenz: Predictions berechnen und dumpen
# =============================================================================
def inference(sceneIds):
    """
    Läuft das Modell auf allen Test-Frames und speichert pro Frame eine
    NPY-Datei mit den vorhergesagten Grasps (im GraspNet-Format).
    """
    test_dataset = GraspnetPointDataset(args.all_points_num,
                                        args.dataset_path,
                                        args.scene_path,
                                        sceneIds,
                                        sigma=args.sigma,
                                        ratio=args.ratio,
                                        anchor_k=args.anchor_k,
                                        anchor_z=args.anchor_z,
                                        anchor_w=args.anchor_w,
                                        grasp_count=args.grasp_count,
                                        output_size=(args.input_w,
                                                     args.input_h),
                                        random_rotate=False,
                                        random_zoom=False,
                                        frames_per_scene=args.frames_per_scene)

    SCENE_LIST = test_dataset.scene_list()
    test_data = DataLoader(test_dataset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=1)
    test_data.dataset.unaug()
    test_data.dataset.eval()

    # Modelle initialisieren
    input_channels = 4
    anchornet = AnchorGraspNet(in_dim=input_channels,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k).cuda()
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2).cuda()

    # Checkpoint laden
    try:
        check_point = torch.load(args.checkpoint_path, weights_only=False)
    except TypeError:
        check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])

    # Anchors aus Checkpoint übernehmen
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    if 'gamma' in check_point and len(check_point['gamma']) == args.anchor_num:
        anchors['gamma'] = check_point['gamma']
        anchors['beta'] = check_point['beta']
        logging.info('Using saved anchors from checkpoint')
    logging.info(f'-> loaded checkpoint {args.checkpoint_path}')

    anchornet.eval()
    localnet.eval()
    test_dataset.eval()

    # Visualisierungs-Frames
    vis_id = []
    if args.vis_frame_ids:
        vis_id = [int(x) for x in args.vis_frame_ids.split(',') if x.strip()]

    time_2d, time_data, time_6d, time_colli, time_nms = 0, 0, 0, 0, 0

    # Sammelt alle Predictions für Evaluation
    all_predictions = []   # list of (idx, grasppath, GraspNetGraspGroup)

    batch_idx = -1
    with torch.no_grad():
        for anchor_data, rgb, ori_depth, grasppaths in test_data:
            batch_idx += 1

            # Depth verarbeiten (medfilt optional, hier deaktiviert)
            depth = ori_depth.numpy().squeeze()
            depth = torch.from_numpy(depth).float()[None]

            # Punktwolke aus Depth
            view_points, _, _ = test_data.dataset.helper.to_scene_points(
                rgb.cuda(), depth.cuda(), include_rgb=True)
            points = view_points[..., :3]
            view_points = view_points.squeeze()
            xyzs = test_data.dataset.helper.to_xyz_maps(depth.cuda())

            # GT-Labels (nur für Eval später)
            all_grasp_labels = []
            for grasppath in grasppaths:
                all_grasp_labels.append(np.load(grasppath))

            # ---------------- 2D-Vorhersage ----------------
            start = time()
            x, _, _, _, _ = anchor_data
            x = x.cuda(non_blocking=True)
            pred_2d, perpoint_features = anchornet(x)

            loc_map, cls_mask, theta_offset, height_offset, width_offset = \
                anchor_output_process(*pred_2d, sigma=args.sigma)

            rect_gg = detect_2d_grasp(loc_map,
                                      cls_mask,
                                      theta_offset,
                                      height_offset,
                                      width_offset,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_w=args.anchor_w,
                                      anchor_z=args.anchor_z,
                                      mask_thre=args.heatmap_thres,
                                      center_num=args.center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size,
                                      reduce='max')

            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_2d += time() - start

            # 2D-Bbox-Validierung (für Eval)
            idx = anchor_data[2].numpy().squeeze()
            grasp_label = test_data.dataset.load_grasp_labels(idx)
            gt_rect_gg = RectGraspGroup()
            gt_rect_gg.load_from_dict(grasp_label)

            if rect_gg.size == 0:
                print(f'[{batch_idx}] No 2d grasp found, skipping')
                # Trotzdem leeres Ergebnis dumpen, damit alle Frames vorkommen
                gg_empty = GraspNetGraspGroup()
                _save_predictions(args, batch_idx, SCENE_LIST, gg_empty)
                all_predictions.append({
                    'batch_idx': batch_idx,
                    'idx': int(idx),
                    'grasppath': grasppaths[0],
                    'rect_gg': rect_gg,
                    'gt_rect_gg': gt_rect_gg,
                    'pred_grasp': None,
                })
                continue

            # ---------------- Feature-Fusion + PC-Crop ----------------
            start = time()
            points_all = feature_fusion(points, perpoint_features, xyzs)
            rect_ggs = [rect_gg]
            pc_group, valid_local_centers = data_process(
                points_all,
                depth.cuda(),
                rect_ggs,
                args.center_num,
                args.group_num, (args.input_w, args.input_h),
                min_points=32,
                is_training=False)
            rect_gg = rect_ggs[0]
            points_all = points_all.squeeze()

            grasp_info = np.zeros((0, 3), dtype=np.float32)
            g_thetas = rect_gg.thetas[None]
            g_ws = rect_gg.widths[None]
            g_ds = rect_gg.depths[None]
            cur_info = np.vstack([g_thetas, g_ws, g_ds])
            grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                         device='cuda')

            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_data += time() - start

            # ---------------- 6D-Vorhersage ----------------
            start = time()
            _, pred, offset = localnet(pc_group, grasp_info)

            pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg,
                pred,
                offset,
                valid_local_centers, (args.input_w, args.input_h),
                anchors,
                k=args.local_k)

            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_6d += time() - start

            # ---------------- Kollisionserkennung ----------------
            start = time()
            pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
            pred_gg, valid_mask = collision_detect(points_all,
                                                   pred_grasp_from_rect,
                                                   mode='graspnet')
            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_colli += time() - start

            # ---------------- NMS + Sortierung ----------------
            start = time()
            gg = GraspNetGraspGroup()
            for pred_g in pred_gg:
                g = GraspNetGrasp(pred_g.score, pred_g.width, pred_g.height,
                                  pred_g.depth, pred_g.rotation.reshape(9,),
                                  pred_g.translation, -1)
                gg.add(g)
            gg = gg.nms(0.03, 30 / 180 * np.pi)
            gg = gg.sort_by_score()

            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_nms += time() - start

            # Optional: Open3D-Visualisierung für ausgewählte Frames
            if batch_idx in vis_id:
                pred_gg_viz = HGGDGraspGroup(translations=gg.translations,
                                              rotations=gg.rotation_matrices,
                                              heights=gg.heights,
                                              widths=gg.widths,
                                              depths=gg.depths,
                                              scores=gg.scores,
                                              object_ids=gg.object_ids)
                print(f'[{batch_idx}] viz: {len(pred_gg_viz)} grasps')
                grasp_geo = pred_gg_viz.to_open3d_geometry_list()
                cloud = view_points[:, :3].cpu().numpy()
                rgb_np = view_points[:, 3:6].cpu().numpy()
                vispc = o3d.geometry.PointCloud()
                vispc.points = o3d.utility.Vector3dVector(cloud)
                vispc.colors = o3d.utility.Vector3dVector(rgb_np)
                o3d.visualization.draw_geometries([vispc] + grasp_geo)

            log_str = (f'[{batch_idx:4d}] {SCENE_LIST[batch_idx]} '
                       f'pred-2d: {rect_gg.size}  pred-6d: {len(pred_grasp)}  '
                       f'no-colli: {valid_mask.sum()}  nms: {len(gg)}')
            print(log_str)

            # Predictions auf Festplatte dumpen
            _save_predictions(args, batch_idx, SCENE_LIST, gg)

            # Für Eval merken
            pred_grasp_tensor = torch.from_numpy(pred_grasp).to(
                device='cuda', dtype=torch.float32)
            all_predictions.append({
                'batch_idx': batch_idx,
                'idx': int(idx),
                'grasppath': grasppaths[0],
                'rect_gg': rect_gg,
                'gt_rect_gg': gt_rect_gg,
                'pred_grasp': pred_grasp_tensor,
            })

    # Time-Statistik
    if batch_idx > 1:
        denom = batch_idx - 1
        time_2d_ms = time_2d / denom * 1000
        time_data_ms = time_data / denom * 1000
        time_6d_ms = time_6d / denom * 1000
        time_colli_ms = time_colli / denom * 1000
        time_nms_ms = time_nms / denom * 1000
        total_ms = time_2d_ms + time_data_ms + time_6d_ms + time_colli_ms + time_nms_ms
        logging.info('Time stats (per frame, averaged):')
        logging.info(f'  Total: {total_ms:.2f} ms')
        logging.info(f'  2D:     {time_2d_ms:.2f} ms')
        logging.info(f'  Data:   {time_data_ms:.2f} ms')
        logging.info(f'  6D:     {time_6d_ms:.2f} ms')
        logging.info(f'  Colli:  {time_colli_ms:.2f} ms')
        logging.info(f'  NMS:    {time_nms_ms:.2f} ms')

    return all_predictions, test_dataset


def _save_predictions(args, batch_idx, scene_list, gg):
    """Speichert die Predictions als NPY-Datei pro Frame."""
    save_dir = os.path.join(args.dump_dir, scene_list[batch_idx], camera)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_idx = batch_idx % args.frames_per_scene
    save_path = os.path.join(save_dir, f'{frame_idx:04d}.npy')
    gg.save_npy(save_path)


# =============================================================================
# Eigene Evaluation: NPZ-Labels vs. Predictions
# =============================================================================
def evaluate(all_predictions, dataset):
    """
    Vergleicht alle Predictions mit GT-Labels aus den NPZ-Files.
    Gibt aggregierte Metriken aus (analog zu validate() in train_graspnet.py).

    Args:
        all_predictions: List der Predictions von inference()
        dataset: Das Test-Dataset-Objekt (für get_gtbb())
    """
    if not all_predictions:
        logging.warning('No predictions to evaluate.')
        return

    results = {
        'iou_correct': 0,
        'iou_total': 0,
        'cover_cnt': 0,
        'label_cnt': 0,
    }
    for scale_factor in eval_scale:
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor
        results[f'grasp_{scale_factor}'] = np.zeros((2,))
        results[thre_key('trans', thre_dis)] = np.zeros((2,))
        results[thre_key('rot', thre_rot)] = np.zeros((2,))

    n_skipped = 0

    for entry in all_predictions:
        if entry['pred_grasp'] is None:
            n_skipped += 1
            continue

        rect_gg = entry['rect_gg']
        gt_rect_gg = entry['gt_rect_gg']
        pred_grasp = entry['pred_grasp']
        grasppath = entry['grasppath']

        # 2D-IoU-Match (rotation/zoom = 0/1 für Eval-Modus)
        rot = 0.0
        zoom_factor = 1.0
        try:
            # GT-Bboxen über die Dataset-Methode holen (macht Skalierung korrekt)
            gt_bbs = dataset.get_gtbb(gt_rect_gg, rot, zoom_factor)
            s = calculate_iou_match(rect_gg[0:args.iou_topk], gt_bbs, thre=args.iou_thre)
            if s:
                results['iou_correct'] += 1
            results['iou_total'] += 1
        except Exception as e:
            logging.warning(f'IoU calc failed for batch {entry["batch_idx"]}: {e}')

        # 6D-Match gegen Original-Labels
        try:
            gg_ori_labels = get_ori_grasp_label([grasppath])
            for scale_factor in eval_scale:
                thre_dis = dis_criterion * scale_factor
                thre_rot = rot_criterion * scale_factor
                r_g, r_d, r_r = calculate_6d_match(pred_grasp,
                                                   gg_ori_labels,
                                                   threshold_dis=thre_dis,
                                                   threshold_rot=thre_rot)
                results[f'grasp_{scale_factor}'] += r_g
                results[thre_key('trans', thre_dis)] += r_d
                results[thre_key('rot', thre_rot)] += r_r

            cover_cnt = calculate_coverage(pred_grasp, gg_ori_labels,
                                           threshold_dis=args.coverage_dis,
                                           threshold_rot=args.coverage_rot)
            results['cover_cnt'] += cover_cnt
            results['label_cnt'] += len(gg_ori_labels)
        except Exception as e:
            logging.warning(f'6D match failed for batch {entry["batch_idx"]}: {e}')

    # Ergebnisse loggen
    logging.info('=' * 60)
    logging.info('TEST EVALUATION RESULTS')
    logging.info('=' * 60)
    logging.info(f'Total frames evaluated: {len(all_predictions)} '
                 f'(skipped: {n_skipped})')

    if results['iou_total'] > 0:
        iou_rate = results['iou_correct'] / results['iou_total']
        logging.info(f'2D IoU@0.25: {results["iou_correct"]}/{results["iou_total"]} '
                     f'= {iou_rate:.4f}')

    logging.info('-' * 60)
    logging.info('6D Grasp Matches (different scale thresholds):')
    logging.info('NOTE: trans/rot/grasp are bidirectional matches (pred->gt and gt->pred).')
    logging.info('Format: right / all candidates = rate')
    for scale_factor in eval_scale:
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor
        g_arr = results[f'grasp_{scale_factor}']
        d_arr = results[thre_key('trans', thre_dis)]
        r_arr = results[thre_key('rot', thre_rot)]
        g_ok, g_bad = int(g_arr[0]), int(g_arr[1])
        d_ok, d_bad = int(d_arr[0]), int(d_arr[1])
        r_ok, r_bad = int(r_arr[0]), int(r_arr[1])
        logging.info(
            f'  scale={scale_factor:.2f}  '
            f'trans_{thre_dis:.2f}: {d_ok}/{d_ok + d_bad} = {d_ok / max(d_ok + d_bad, 1):.3f}  '
            f'rot_{thre_rot:.2f}: {r_ok}/{r_ok + r_bad} = {r_ok / max(r_ok + r_bad, 1):.3f}  '
            f'grasp_{scale_factor:.2f}: {g_ok}/{g_ok + g_bad} = {g_ok / max(g_ok + g_bad, 1):.3f}')
        

    logging.info('-' * 60)
    if results['label_cnt'] > 0:
        cov_rate = results['cover_cnt'] / results['label_cnt']
        logging.info(f'Coverage: {results["cover_cnt"]} / {results["label_cnt"]} '
                     f'= {cov_rate:.4f}')
    logging.info('=' * 60)
    return results


def _make_gt_bbs(gt_rect_gg):
    """[deprecated, kept for compatibility] Use dataset.get_gtbb() instead."""
    return gt_rect_gg

def _metrics_to_json(results):
    """calculate_6d_match liefert [richtig, falsch] -- NICHT [matched, total].
    Der Nenner ist die Summe beider, nicht das zweite Element."""
    def as_array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    out, seen_n = {}, {}
    for key, value in results.items():
        arr = as_array(value)
        if arr.size == 2:
            correct, false = float(arr[0]), float(arr[1])
            n = correct + false
            out[key] = {'correct': correct, 'false': false, 'n': n,
                        'rate': correct / max(n, 1.0)}
            seen_n[key] = n
        else:
            out[key] = float(arr.reshape(-1)[0])

    # Selbsttest: alle Schwellenmetriken zaehlen dieselbe Grundmenge.
    # Weichen sie ab, ist die Akkumulation kaputt -- lieber hier auffallen.
    if len(set(seen_n.values())) > 1:
        logging.warning(f'Grundmenge nicht einheitlich: {seen_n}')

    out['thresholds'] = {'dis_criterion': dis_criterion,
                         'rot_criterion': rot_criterion,
                         'eval_scale': list(eval_scale)}
    out['rates'] = {
        'iou': out['iou_correct'] / max(out['iou_total'], 1.0),
        'coverage': out['cover_cnt'] / max(out['label_cnt'], 1.0),
        'grasp_full': out[f'grasp_{eval_scale[-1]}']['rate'],
    }
    out['constants'] = dict(EVAL_CONSTANTS)
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Output-Verzeichnis vorbereiten
    net_desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = f'graspnet_test_{net_desc}_{args.description}'

    save_folder = os.path.join(args.logdir, net_desc)
    os.makedirs(save_folder, exist_ok=True)

    params_path = os.path.join(save_folder, 'commandline_args.json')
    with open(params_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_folder, 'log.log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)

    logging.info('=' * 60)
    logging.info('VIRAS HGGD Test')
    logging.info('=' * 60)
    logging.info(f'Checkpoint:    {args.checkpoint_path}')
    logging.info(f'Dataset path:  {args.dataset_path}')
    logging.info(f'Scene path:    {args.scene_path}')
    logging.info(f'Frames/scene:  {args.frames_per_scene}')
    logging.info(f'Dump dir:      {args.dump_dir}')

    # Test-Szenen laden
    sceneIds = load_test_scenes(args.scene_path)
    if not sceneIds:
        logging.error('No test scenes found, aborting.')
        return

    # Inferenz
    all_predictions, test_dataset = inference(sceneIds)

    # Evaluation (wenn gewünscht)
    if not args.skip_eval:
        results = evaluate(all_predictions, test_dataset)
        if results is not None:
            metrics_path = os.path.join(save_folder, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(_metrics_to_json(results), f, indent=2)
            logging.info(f'metrics.json geschrieben: {metrics_path}')
    else:
        logging.info('Evaluation skipped (--skip-eval).')


if __name__ == '__main__':
    main()