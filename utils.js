const posenet = require('@tensorflow-models/posenet');
const minPoseConfidence = 0.5;

const drawKeypoints = (keypoints, ctx, scale = 1) => {
    const radius = 2 * scale;
    keypoints.forEach(keypoint => {
      if (keypoint.score >= minPoseConfidence) {
        const {y, x} = keypoint.position;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = 'aqua';
        ctx.fill();
      }
    });
  };
  
const drawSkeleton = (keypoints, ctx, scale = 1) => {
    const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minPoseConfidence);

    adjacentKeyPoints.forEach(keypoints => {
        const fromPosition = keypoints[0].position;
        const toPosition = keypoints[1].position;

        ctx.beginPath();
        ctx.moveTo(fromPosition.x, fromPosition.y);
        ctx.lineTo(toPosition.x, toPosition.y);
        ctx.lineWidth = 2 * scale;
        ctx.strokeStyle = 'aqua';
        ctx.stroke();
    });
};

module.exports = { drawKeypoints, drawSkeleton };