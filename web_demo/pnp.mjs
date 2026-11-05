/** GPL-3.0-only. C4 pose fitting through the official OpenCV.js WASM solver.
 * Fitting is unweighted in this build. Point confidence/visibility select the
 * usable set and affect acceptance diagnostics, but do not weight refinement.
 */
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function scaleGeometry(geometry, heightM, baseWidthM = null) {
  if (!(heightM > 0)) throw Error('A measured positive cone height is required for metric output');
  const sz = heightM / geometry.height_m;
  const sxy = baseWidthM === null ? sz : baseWidthM / geometry.base_width_m;
  if (!(sxy > 0)) throw Error('Base width must be positive');
  return geometry.keypoints_3d.map(p => [p[0] * sxy, p[1] * sxy, p[2] * sz]);
}

export function poseSettingsForRoi(roi) {
  const scale = Math.max(1, roi.side / (roi.size ?? 96));
  return {measurementSigmaPx: 1.5*scale, maxReprojectionPx: 6*scale};
}

export async function loadOpenCV(url = './vendor/opencv.js') {
  if (!globalThis.cv) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url; script.onload = resolve;
      script.onerror = () => reject(Error('Could not load the bundled OpenCV.js runtime'));
      document.head.appendChild(script);
    });
  }
  let cv = globalThis.cv;
  if (cv instanceof Promise) cv = await cv;
  if (!cv.Mat) await new Promise((resolve, reject) => {
    const previous = cv.onRuntimeInitialized;
    cv.onRuntimeInitialized = () => { if (previous) previous(); resolve(); };
    cv.onAbort = reject;
  });
  if (typeof cv.solvePnP !== 'function' || typeof cv.Rodrigues !== 'function') {
    throw Error('This OpenCV.js build lacks the required pose functions');
  }
  // The official runtime exposes a legacy thenable that resolves to itself.
  // Wrap it so async return-value assimilation does not recurse indefinitely.
  return {cv};
}

export function solveConePose(cv, points3d, points2d, K, confidence = null,
                              visibility = null, distortion = [], settings = {}) {
  const measurementSigmaPx = settings.measurementSigmaPx ?? 1.5;
  const maxReprojectionPx = settings.maxReprojectionPx ?? 6;
  if (points3d.length !== 5 || points2d.length !== 5) throw Error('Exactly five ordered anchors are required');
  if (!Array.isArray(K) || K.length !== 3 || K.some(row => row.length !== 3) ||
      !K.flat().every(Number.isFinite) || !(K[0][0] > 0) || !(K[1][1] > 0) ||
      Math.abs(K[0][1]) > 1e-9 || Math.abs(K[1][0]) > 1e-9 ||
      K[2][0] !== 0 || K[2][1] !== 0 || K[2][2] !== 1) throw Error('Invalid zero-skew camera matrix');
  const conf = confidence ?? [1, 1, 1, 1, 1];
  const probabilities = visibility ?? Array.from({length: 5}, () => [0, 0, 1]);
  const used = points2d.map((p, i) => p.every(Number.isFinite) && conf[i] >= 0.05 &&
    probabilities[i][1] + probabilities[i][2] >= 0.25);
  const visibleSupport = used.filter((v, i) => v && probabilities[i][2] >= 0.5).length;
  const failure = {success: false, accepted: false, yaw_observable: false, symmetry: 'C4', candidates: [],
    reason: '', usable_points: used.filter(Boolean).length, backend: 'OpenCV.js 4.13.0 WASM'};
  if (failure.usable_points < 4) return {...failure, reason: 'fewer_than_four_usable_points'};
  const subset = points2d.filter((_, i) => used[i]);
  const width = Math.max(...subset.map(p => p[0])) - Math.min(...subset.map(p => p[0]));
  const height = Math.max(...subset.map(p => p[1])) - Math.min(...subset.map(p => p[1]));
  const extent = Math.hypot(width, height);
  let maxTriangleArea = 0;
  for (let a = 0; a < subset.length; a++) for (let b = a + 1; b < subset.length; b++)
    for (let c = b + 1; c < subset.length; c++) {
      const p = subset[a], q = subset[b], r = subset[c];
      maxTriangleArea = Math.max(maxTriangleArea, Math.abs((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])) / 2);
    }
  if (extent < 4 || maxTriangleArea < 0.002 * extent * extent)
    return {...failure, reason: 'image_points_too_small_or_collinear'};

  const camera = cv.matFromArray(3, 3, cv.CV_64F, K.flat());
  const dist = distortion.length ? cv.matFromArray(distortion.length, 1, cv.CV_64F, distortion) : cv.Mat.zeros(5, 1, cv.CV_64F);
  const allObj = cv.matFromArray(5, 3, cv.CV_64F, points3d.flat());
  const candidates = [];
  try {
    for (let shift = 0; shift < 4; shift++) {
      const permutation = [0, ...[0, 1, 2, 3].map(j => 1 + (j + shift) % 4)];
      const ids = [0, 1, 2, 3, 4].filter(i => used[permutation[i]]);
      const isPlanar = !ids.includes(0);
      const obj = cv.matFromArray(ids.length, 3, cv.CV_64F, ids.flatMap(i => points3d[i]));
      const img = cv.matFromArray(ids.length, 2, cv.CV_64F, ids.flatMap(i => points2d[permutation[i]]));
      const rvec = new cv.Mat(), tvec = new cv.Mat(), rotation = new cv.Mat(), projected = new cv.Mat();
      const jacobian = new cv.Mat(), normal = cv.Mat.zeros(6, 6, cv.CV_64F), inverse = new cv.Mat();
      try {
        let ok = false;
        const methods = isPlanar ? [cv.SOLVEPNP_IPPE, cv.SOLVEPNP_SQPNP] : [cv.SOLVEPNP_SQPNP, cv.SOLVEPNP_EPNP];
        for (const flag of methods) {
          if (typeof flag !== 'number') continue;
          try { ok = cv.solvePnP(obj, img, camera, dist, rvec, tvec, false, flag); } catch { ok = false; }
          if (ok) break;
        }
        if (!ok) continue;
        // Valid nonlinear initialization is crucial: five non-planar points are
        // insufficient for the default six-point DLT initializer.
        cv.solvePnP(obj, img, camera, dist, rvec, tvec, true, cv.SOLVEPNP_ITERATIVE);
        cv.Rodrigues(rvec, rotation);
        const t = Array.from(tvec.data64F), flatR = Array.from(rotation.data64F);
        const R = [flatR.slice(0, 3), flatR.slice(3, 6), flatR.slice(6, 9)];
        const depths = points3d.map(p => R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2] + t[2]);
        if (![...t, ...flatR].every(Number.isFinite) || Math.min(...depths) <= 1e-4 || t[2] <= 0) continue;
        cv.projectPoints(allObj, rvec, tvec, camera, dist, projected, jacobian);
        const xy = Array.from({length: 5}, (_, i) => [projected.data64F[2*i], projected.data64F[2*i+1]]);
        let sumSq = 0, weighted = 0, sumWeight = 0;
        for (const id of ids) {
          const prediction = permutation[id], observed = points2d[prediction];
          const sq = (xy[id][0]-observed[0])**2 + (xy[id][1]-observed[1])**2;
          const v = probabilities[prediction], w = clip(conf[prediction], 0, 1)**2 * (v[2] + .35*v[1]);
          sumSq += sq; weighted += w*sq; sumWeight += w;
          for (let row = 2*id; row < 2*id+2; row++) for (let a = 0; a < 6; a++)
            for (let b = 0; b < 6; b++) normal.data64F[a*6+b] +=
              w*jacobian.data64F[row*jacobian.cols+a]*jacobian.data64F[row*jacobian.cols+b];
        }
        const weightedRmse = Math.sqrt(weighted/Math.max(sumWeight, 1e-12));
        // OpenCV invert(DECOMP_SVD) returns the reciprocal condition number.
        // A failed/rank-deficient information matrix cannot imply zero error.
        const reciprocalCondition = cv.invert(normal, inverse, cv.DECOMP_SVD);
        const conditioned = Number.isFinite(reciprocalCondition) && reciprocalCondition > 1e-12;
        const sigma = Math.max(measurementSigmaPx, weightedRmse);
        const depthSigma = conditioned ? Math.sqrt(Math.max(0, inverse.data64F[35]))*sigma : null;
        candidates.push({cyclic_shift: shift, prediction_indices_for_3d_ids: permutation,
          R, t, rvec: Array.from(rvec.data64F), depth_m: t[2], distance_m: Math.hypot(...t),
          reprojection_rmse_px: Math.sqrt(sumSq/ids.length),
          weighted_reprojection_rmse_px: weightedRmse,
          information_reciprocal_condition: reciprocalCondition, local_depth_sigma_m: depthSigma,
          local_relative_depth_sigma: depthSigma === null ? null : depthSigma/t[2],
          reprojected_keypoints_2d: xy, coplanar: isPlanar, fit_point_ids: ids});
      } catch { /* A failed cyclic candidate is omitted, never silently invented. */ }
      finally { for (const mat of [obj, img, rvec, tvec, rotation, projected, jacobian, normal, inverse]) mat.delete(); }
    }
  } finally { camera.delete(); dist.delete(); allObj.delete(); }
  if (!candidates.length) return {...failure, reason: 'no_positive_depth_pose'};
  candidates.sort((a,b) => a.reprojection_rmse_px-b.reprojection_rmse_px);
  const best = candidates[0];
  const accepted = !best.coplanar && failure.usable_points === 5 && extent >= 24 && visibleSupport >= 2 &&
    best.reprojection_rmse_px <= maxReprojectionPx && best.reprojection_rmse_px / extent <= .05 &&
    best.local_relative_depth_sigma !== null && best.local_relative_depth_sigma <= .3;
  const warnings = ['absolute_yaw_ambiguous_modulo_90_degrees', 'metric_scale_and_intrinsics_must_be_calibrated',
    'browser_refinement_is_unweighted', 'no_calibrated_distance_confidence_interval'];
  if (used.some((v,i) => v && probabilities[i][2] < .5)) warnings.push('uses_inferred_occluded_points');
  if (best.coplanar) warnings.push('planar_pose_solutions_not_exhaustively_enumerated');
  if (failure.usable_points < 5) warnings.push('browser_requires_five_points_for_acceptance');
  if (extent < 24) warnings.push('too_small_for_reliable_metric_output');
  if (best.local_relative_depth_sigma === null) warnings.push('depth_conditioning_unavailable_or_rank_deficient');
  else if (best.local_relative_depth_sigma > .3) warnings.push('poor_depth_conditioning');
  if (!accepted) warnings.push('low_reliability');
  return {...best, success: true, accepted, reason: accepted ? 'ok' : 'low_reliability',
    yaw_observable: false, symmetry: 'C4', candidates, correspondence_candidates: candidates,
    image_extent_px: extent, usable_points: failure.usable_points, predicted_visible_support: visibleSupport,
    acceptance_settings: {measurement_sigma_px: measurementSigmaPx, max_reprojection_px: maxReprojectionPx,
      max_relative_reprojection: .05, min_accepted_extent_px: 24},
    backend: 'OpenCV.js 4.13.0 WASM', warnings,
    uncertainty_note: 'Residual/extent/local-Jacobian filters are engineering heuristics. They exclude size/calibration errors and correlated hidden-point bias. Python uses weighted robust refinement; this browser build fits points without weights.'};
}
