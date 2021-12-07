const apiurl = 'https://itc.gymkhana.iitb.ac.in/tcb_backend';

function capture() {
	const cnv = document.getElementById("canvasInput");
	cnv.width = video.videoWidth;
	cnv.height = video.videoHeight;
	
	const ctx = cnv.getContext('2d');
	ctx.drawImage(video, 0, 0);
}

function loadOpenCv(onloadCallback) {
	const OPENCV_URL = 'opencv.js';
	let script = document.createElement('script');
	script.setAttribute('async', '');
	script.setAttribute('type', 'text/javascript');
	script.addEventListener('load', async () => {
		if (cv.getBuildInformation) {
			console.log(cv.getBuildInformation());
			onloadCallback();
		} else {
			// WASM
			if (cv instanceof Promise) {
				cv = await cv;
				console.log(cv.getBuildInformation());
				onloadCallback();
			} else {
				cv['onRuntimeInitialized'] = () => {
					console.log(cv.getBuildInformation());
					onloadCallback();
				}
			}
		}
	});
	
	script.addEventListener('error', () => {
		self.printError('Failed to load ' + OPENCV_URL);
	});
	script.src = OPENCV_URL;
	let node = document.getElementsByTagName('script')[0];
	node.parentNode.insertBefore(script, node);
}

function createFileFromUrl(path, url, callback) {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function(ev) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                let data = new Uint8Array(request.response);
                cv.FS_createDataFile('/', path, data, true, false, false);
                callback();
            } else {
                console.log('Failed to load ' + url + ' status: ' + request.status);
            }
        }
    };
    request.send();
}

function detectFace() {
	let src_mat = new cv.Mat();
	let dst_mat = new cv.Mat();
	let gray_mat = new cv.Mat();
	let gray_mat_roi = new cv.Mat();
	let faces_rv = new cv.RectVector();

	src_mat = cv.imread(document.getElementById('canvasInput'));
	src_mat.copyTo(dst_mat);
	cv.cvtColor(dst_mat, gray_mat, cv.COLOR_RGBA2GRAY);

	let cc = new cv.CascadeClassifier();
	cc.load('haarcascade_frontalface_alt2.xml');
	cc.detectMultiScale(gray_mat, faces_rv);

	let face_rect = faces_rv.get(0);
	gray_mat_roi = gray_mat.roi(face_rect);
	cv.resize(gray_mat_roi, dst_mat, new cv.Size(48, 48));


	cv.imshow('canvasOutput', dst_mat);
	let retdata = dst_mat.data;

	src_mat.delete();
	dst_mat.delete();
	gray_mat.delete();
	gray_mat_roi.delete();
	faces_rv.delete();
	cc.delete();

	return retdata;
}

function argmax(arr) {
	let am = 0;
	for (i = 0; i < arr.length; i++) {
		if (arr[i] > arr[am])
			am = i;
	}

	return am;
}

var prevam = -1;
async function startlogic(mod) {
	console.log('here');
	capture();
	let roi_mat_data = detectFace();
	let norm_data = new Float32Array(48*48);
	for (let i = 0; i < 48*48; i++) {
		norm_data[i] = roi_mat_data[i] / 255;
	}

	let input_tensor = tf.tensor4d(norm_data, [1, 48, 48, 1]);
	let x = mod.predict(input_tensor);
	let res = x.dataSync();
	let acc = Math.max(...res);
	let am = argmax(res);

	if (prevam === -1) {
		await fetch(apiurl + '/inc/' + am, {method: 'POST'});
	}
	if (prevam != am) {
		await fetch(apiurl + '/dec/' + prevam, {method: 'POST'});
		await fetch(apiurl + '/inc/' + am, {method: 'POST'});
	}
	prevam = am;

	document.getElementById("acc").innerHTML = "acc = " + acc;
	document.getElementById("am").innerHTML = "am = " + am;

	input_tensor.dispose();
	x.dispose();
}

async function endlogic() {
	if (prevam !== -1)
		await fetch(apiurl + '/dec/' + prevam, {method: 'POST'});
}
