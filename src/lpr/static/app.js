const fileInput = document.querySelector('#fileInput');
const dropZone = document.querySelector('#dropZone');
const canvas = document.querySelector('#previewCanvas');
const placeholder = document.querySelector('#dropPlaceholder');
const clearFileButton = document.querySelector('#clearFile');
const predictButton = document.querySelector('#predictButton');
const inputHint = document.querySelector('#inputHint');
const resultEmpty = document.querySelector('#resultEmpty');
const resultContent = document.querySelector('#resultContent');
const resultLabel = document.querySelector('#resultLabel');
const plateResult = document.querySelector('#plateResult');
const resultMeta = document.querySelector('#resultMeta');

let selectedFile = null;
let previewImage = null;
let previewUrl = null;
let modelStatus = null;
let busy = false;
let resultBox = null;

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function drawPreview() {
  if (!previewImage) return;
  canvas.width = previewImage.naturalWidth;
  canvas.height = previewImage.naturalHeight;
  const context = canvas.getContext('2d');
  context.drawImage(previewImage, 0, 0);
  if (resultBox) {
    const [x1, y1, x2, y2] = resultBox;
    const width = Math.max(2, Math.round(canvas.width / 250));
    context.strokeStyle = '#57e88d';
    context.lineWidth = width;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
}

function showMessage(title, detail) {
  resultContent.hidden = true;
  resultEmpty.hidden = false;
  resultEmpty.querySelector('strong').textContent = title;
  resultEmpty.querySelector('p').textContent = detail;
}

function updateAction() {
  const mode = currentMode();
  const recognizerReady = !!modelStatus?.recognizer_ready;
  const detectorReady = !!modelStatus?.detector_ready;
  const ready = recognizerReady && (mode === 'crop' || detectorReady);
  predictButton.disabled = busy || !selectedFile || !ready;
  predictButton.innerHTML = busy ? '正在识别… <span aria-hidden="true">◌</span>' : ready ? '开始识别 <span aria-hidden="true">→</span>' : !recognizerReady ? '请先训练识字模型 <span aria-hidden="true">→</span>' : '请先训练定位模型 <span aria-hidden="true">→</span>';
  if (!selectedFile && ready) predictButton.innerHTML = '请先选择图片 <span aria-hidden="true">→</span>';
  inputHint.textContent = !recognizerReady ? '尚未发现 outputs/recognizer.pt。可先按 README 训练识字模型。' : mode === 'full' && !detectorReady ? '完整车辆图片还需要 outputs/detector.pt；裁剪车牌模式已可使用。' : mode === 'full' ? '图片仅发送到当前电脑上的本地服务，不会保存。' : '裁剪车牌模式直接识别整张上传图片。';
}

async function refreshStatus() {
  const recognizer = document.querySelector('#recognizerStatus');
  const detector = document.querySelector('#detectorStatus');
  const hint = document.querySelector('#statusHint');
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('服务状态不可用');
    modelStatus = await response.json();
    recognizer.textContent = `识字模型：${modelStatus.recognizer_ready ? '已就绪' : '未准备'}`;
    detector.textContent = `定位模型：${modelStatus.detector_ready ? '已就绪' : '未准备'}`;
    recognizer.className = `status-chip ${modelStatus.recognizer_ready ? 'ready' : 'missing'}`;
    detector.className = `status-chip ${modelStatus.detector_ready ? 'ready' : 'missing'}`;
    hint.textContent = modelStatus.recognizer_error || modelStatus.detector_error || (modelStatus.recognizer_ready && modelStatus.detector_ready ? '现在可以识别完整车辆图片。' : '模型文件训练完成后会自动加载。');
  } catch (error) {
    modelStatus = null;
    recognizer.textContent = '识字模型：连接中断';
    detector.textContent = '定位模型：连接中断';
    recognizer.className = detector.className = 'status-chip missing';
    hint.textContent = '无法连接本地服务，请检查终端。';
  }
  updateAction();
}

function chooseFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showMessage('图片格式不支持', '请选择 JPG、PNG 或 WebP 图片。');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showMessage('图片太大', '请选择 10 MB 以内的图片。');
    return;
  }
  selectedFile = file;
  resultBox = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage = new Image();
  previewImage.onload = () => {
    placeholder.hidden = true;
    canvas.hidden = false;
    drawPreview();
  };
  previewImage.onerror = () => showMessage('图片无法打开', '请换一张有效图片。');
  previewImage.src = previewUrl;
  document.querySelector('#fileName').textContent = file.name;
  clearFileButton.hidden = false;
  showMessage('图片已就绪', '点击“开始识别”查看模型预测。');
  updateAction();
}

function clearFile() {
  selectedFile = null;
  previewImage = null;
  resultBox = null;
  fileInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  placeholder.hidden = false;
  canvas.hidden = true;
  clearFileButton.hidden = true;
  document.querySelector('#fileName').textContent = '尚未选择图片';
  showMessage('等待图像输入', '完成识别后，这里会显示车牌号码和定位结果。');
  updateAction();
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); } });
dropZone.addEventListener('dragover', event => { event.preventDefault(); dropZone.classList.add('dragging'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', event => { event.preventDefault(); dropZone.classList.remove('dragging'); chooseFile(event.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => chooseFile(fileInput.files[0]));
clearFileButton.addEventListener('click', clearFile);
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', () => { resultBox = null; drawPreview(); updateAction(); showMessage('模式已切换', '点击“开始识别”查看对应模式的预测。'); }));

predictButton.addEventListener('click', async () => {
  if (!selectedFile || busy) return;
  busy = true;
  updateAction();
  showMessage('正在识别', '本地模型正在处理图片，请稍候。');
  try {
    const response = await fetch(`/api/predict?mode=${currentMode()}`, { method: 'POST', headers: { 'Content-Type': selectedFile.type }, body: selectedFile });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '识别请求失败');
    if (result.status === 'no_plate') {
      showMessage('未找到车牌', '检测器没有找到足够可信的车牌位置。可以换一张清晰图片，或检查模型训练效果。');
      resultBox = null;
      drawPreview();
    } else {
      resultBox = result.box;
      drawPreview();
      resultEmpty.hidden = true;
      resultContent.hidden = false;
      resultLabel.textContent = '识别到的车牌';
      plateResult.replaceChildren(...Array.from(result.plate, char => { const span = document.createElement('span'); span.textContent = char; return span; }));
      let location = '裁剪车牌模式 · 已识别上传图片中的字符';
      if (result.box) {
        const detectorDetail = result.detection_source === 'blue_color'
          ? '蓝牌颜色定位（请核对定位框）'
          : `检测分数：${result.detection_score.toFixed(3)}${result.detection_source === 'cnn_blue_refined' ? ' · 蓝色边界修正' : ''}`;
        location = `定位位置：(${result.box[0]}, ${result.box[1]}) 至 (${result.box[2]}, ${result.box[3]}) · ${detectorDetail}`;
      }
      resultMeta.textContent = result.province_uncertain ? `${location} · 省份字把握较低，请核对首字` : location;
    }
  } catch (error) {
    showMessage('暂时无法识别', error.message);
    refreshStatus();
  } finally {
    busy = false;
    updateAction();
  }
});

const inputPixels = [0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1, 0,0,1,1,1];
const kernels = { edge: [-1,0,1,-1,0,1,-1,0,1], blur: [1,1,1,1,1,1,1,1,1] };
let kernelName = 'edge';
let activeOutput = 4;

function renderConvolution() {
  const kernel = kernels[kernelName];
  const output = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let sum = 0;
      for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++) sum += inputPixels[(row + kr) * 5 + col + kc] * kernel[kr * 3 + kc];
      output.push(sum);
    }
  }
  const selectedRow = Math.floor(activeOutput / 3), selectedCol = activeOutput % 3;
  const inputGrid = document.querySelector('#inputGrid');
  inputGrid.replaceChildren(...inputPixels.map((value, index) => {
    const cell = document.createElement('button');
    cell.type = 'button'; cell.textContent = value; cell.className = value ? 'on' : '';
    const row = Math.floor(index / 5), col = index % 5;
    if (row >= selectedRow && row < selectedRow + 3 && col >= selectedCol && col < selectedCol + 3) cell.classList.add('highlight');
    cell.setAttribute('aria-label', `第 ${row + 1} 行第 ${col + 1} 列像素，当前为 ${value}，点击切换`);
    cell.addEventListener('click', () => { inputPixels[index] = value ? 0 : 1; renderConvolution(); });
    return cell;
  }));
  const kernelGrid = document.querySelector('#kernelGrid');
  kernelGrid.replaceChildren(...kernel.map(value => { const cell = document.createElement('span'); cell.textContent = value; return cell; }));
  const outputGrid = document.querySelector('#outputGrid');
  outputGrid.replaceChildren(...output.map((value, index) => {
    const cell = document.createElement('button'); cell.type = 'button'; cell.textContent = value;
    cell.className = index === activeOutput ? 'active' : '';
    cell.setAttribute('aria-label', `输出第 ${Math.floor(index / 3) + 1} 行第 ${index % 3 + 1} 列，值为 ${value}，点击查看对应输入区域`);
    cell.addEventListener('click', () => { activeOutput = index; renderConvolution(); });
    return cell;
  }));
  document.querySelector('#formula').textContent = `当前窗口：第 ${selectedRow + 1} 行、第 ${selectedCol + 1} 列起 · 9 组数字相乘后相加 = ${output[activeOutput]}`;
}

document.querySelectorAll('.kernel-option').forEach(button => button.addEventListener('click', () => {
  kernelName = button.dataset.kernel;
  document.querySelectorAll('.kernel-option').forEach(option => option.classList.toggle('active', option === button));
  renderConvolution();
}));

renderConvolution();
refreshStatus();
setInterval(refreshStatus, 15000);
