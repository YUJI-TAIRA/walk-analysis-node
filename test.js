const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, Image } = require('canvas');
const ffmpeg = require('fluent-ffmpeg');
const { exec } = require('child_process');
const posenet = require('@tensorflow-models/posenet');
const { drawKeypoints, drawSkeleton } = require('./utils');

const videoFile = process.argv[2]; 

if (!fs.existsSync(videoFile)) {
    console.error(`The file ${videoFile} does not exist.`);
    process.exit(1);
}

const now = new Date();
const timestamp = `${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()}_${now.getHours()}-${now.getMinutes()}-${now.getSeconds()}`;

// Define output directories
const outputPath = path.join(__dirname, 'output');
const tmpFramesDir = path.join(outputPath, 'tmp_frames');

// Ensure output directories exist
if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath);
}

if (!fs.existsSync(tmpFramesDir)) {
    fs.mkdirSync(tmpFramesDir);
}

const outputVideoName = `stickman_${timestamp}.mp4`;
const outputVideoPath = path.join(outputPath, outputVideoName);

function getFramesFromVideo(videoFile) {
    return new Promise((resolve, reject) => {
        const frames = [];
        const outputPattern = path.join(tmpFramesDir, 'out-%04d.png');

        const command = `ffmpeg -i ${videoFile} -vf "fps=30" ${outputPattern}`;

        exec(command, (err) => {
            if (err) {
                reject(err);
                return;
            }

            fs.readdir(tmpFramesDir, (err, files) => {
                if (err) {
                    reject(err);
                    return;
                }

                for (let file of files) {
                    const filePath = path.join(tmpFramesDir, file);
                    const data = fs.readFileSync(filePath);
                    frames.push(data);
                    fs.unlinkSync(filePath);
                }

                resolve(frames);
            });
        });
    });
}

async function createStickmanImage(net, imageData, frameIndex) {
    const image = new Image();
    image.src = imageData;
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    const input = tf.browser.fromPixels(canvas);
    const pose = await net.estimateSinglePose(input, {
        flipHorizontal: false,
    });
    input.dispose();
    drawKeypoints(pose.keypoints, ctx); 
    drawSkeleton(pose.keypoints, ctx); 
    const outputFramePath = path.join(tmpFramesDir, `frame-${frameIndex}.png`);
    fs.writeFileSync(outputFramePath, canvas.toBuffer());
    console.log(`The PNG file was created: ${outputFramePath}`);
}

(async () => {
    try {
        const net = await posenet.load(); 
        const frames = await getFramesFromVideo(videoFile);

        for (let i = 0; i < frames.length; i++) {
            console.log(`Processing frame ${i + 1} / ${frames.length}`);
            await createStickmanImage(net, frames[i], i + 1);
        }

        // Generate the video file
        const generateVideoCommand = `ffmpeg -framerate 30 -i ${tmpFramesDir}/frame-%d.png -c:v libx264 -pix_fmt yuv420p ${outputVideoPath}`;
        exec(generateVideoCommand, (err) => {
            if (err) {
                console.error('An error occurred: ' + err.message); // エラーメッセージをコンソールに出力
                return;
            }

            // Remove temporary frame images
            fs.readdir(tmpFramesDir, (err, files) => {
                if (err) {
                    console.error('An error occurred: ' + err.message); // エラーメッセージをコンソールに出力
                    return;
                }

                for (let file of files) {
                    fs.unlinkSync(path.join(tmpFramesDir, file));
                }

                // Remove temporary frame directory
                fs.rmdirSync(tmpFramesDir);
            });

            console.log(`Created ${outputVideoPath}`);
        });

    } catch (err) {
        console.error(err);
    }
})();
