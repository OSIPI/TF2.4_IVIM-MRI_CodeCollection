cornerstoneNIFTIImageLoader.external.cornerstone = cornerstone;
const ImageId = cornerstoneNIFTIImageLoader.nifti.ImageId;
cornerstoneNIFTIImageLoader.nifti.streamingMode = true;
const niftiReader = cornerstoneNIFTIImageLoader.external.niftiReader;

let loaded = false;
const synchronizer = new cornerstoneTools.Synchronizer("cornerstonenewimage", cornerstoneTools.updateImageSynchronizer);

let voxel3dUnits = [0, 0, 0];
let dims = [0, 0, 0, 0];
let niftiImageBuffer = [];

// Helper to calculate voxel position based on view
function updateVoxelCoordinates(view, voxelCoords) {
    if (view === 'axial') {
        voxel3dUnits[0] = voxelCoords.x;
        voxel3dUnits[1] = voxelCoords.y;
    } else if (view === 'sagittal') {
        voxel3dUnits[1] = voxelCoords.x;
        voxel3dUnits[2] = voxelCoords.y;
    } else if (view === 'coronal') {
        voxel3dUnits[0] = voxelCoords.x;
        voxel3dUnits[2] = voxelCoords.y;
    }
}

// Event listener for mouse drag to update voxel coordinates
function addMouseDragListener(element, view) {
    element.addEventListener('cornerstonetoolsmousedrag', (event) => {
        const voxelCoords = event.detail?.currentPoints?.image;
        if (voxelCoords) {
            updateVoxelCoordinates(view, voxelCoords);
            handleVoxelClick(voxel3dUnits);
        }
    });
}

// Load and display NIfTI image on a specific element
function loadAndViewImage(element, imageId, view) {
    const imageIdObject = ImageId.fromURL(imageId);
    element.dataset.imageId = imageIdObject.url;

    cornerstone.loadAndCacheImage(imageIdObject.url).then(image => {
        setupImageViewport(element, image);
        setupImageTools(element, imageIdObject);

        synchronizer.add(element);
        addMouseDragListener(element, view);

        element.addEventListener('cornerstonestackscroll', (event) => updateSliceIndex(view, event.detail.newImageIdIndex));
    }).catch(err => {
        console.error(`Error loading image for ${view} view:`, err);
    });
    element.addEventListener('click', function (event) {
        //TODO: Update the clicked voxel information.
        console.log(event)
    });
}

// Setup viewport and display the image
function setupImageViewport(element, image) {
    const viewport = cornerstone.getDefaultViewportForImage(element, image);
    cornerstone.displayImage(element, image, viewport);
    cornerstone.resize(element, true);
}

// Enable tools and interactions for the displayed image
function setupImageTools(element, imageIdObject) {
    const numberOfSlices = cornerstone.metaData.get('multiFrameModule', imageIdObject.url).numberOfFrames;
    const stack = {
        currentImageIdIndex: imageIdObject.slice.index,
        imageIds: Array.from({ length: numberOfSlices }, (_, i) => `nifti:${imageIdObject.filePath}#${imageIdObject.slice.dimension}-${i},t-0`)
    };

    cornerstoneTools.addStackStateManager(element, ['stack']);
    cornerstoneTools.addToolState(element, 'stack', stack);
    cornerstoneTools.mouseInput.enable(element);
    cornerstoneTools.mouseWheelInput.enable(element);
    cornerstoneTools.pan.activate(element, 2);
    cornerstoneTools.stackScrollWheel.activate(element);
    cornerstoneTools.orientationMarkers.enable(element);
    cornerstoneTools.stackPrefetch.enable(element);
    cornerstoneTools.referenceLines.tool.enable(element, synchronizer);
    cornerstoneTools.crosshairs.enable(element, 1, synchronizer);
}

// Handle voxel click event
function handleVoxelClick(currentVoxel) {
    const [nx, ny, nz, nt] = dims;
    let [voxelX, voxelY, voxelZ] = currentVoxel;

    voxelX = Math.min(Math.max(Math.round(voxelX), 1), nx);
    voxelY = Math.min(Math.max(ny - Math.round(voxelY), 1), ny);
    voxelZ = Math.min(Math.max(nz - Math.round(voxelZ), 1), nz);

    const voxelValues = getVoxelValuesAcrossTime(voxelX, voxelY, voxelZ, nx, ny, nz, nt);
    updateVoxelCoordinatesDisplay(voxelX + 1, voxelY + 1, voxelZ + 1);
    plotVoxelData(voxelValues);
}

// Extract voxel values across all time points
function getVoxelValuesAcrossTime(x, y, z, nx, ny, nz, nt) {
    const sliceSize = nx * ny;
    const volumeSize = sliceSize * nz;
    let voxelValues = [];

    for (let t = 0; t < nt; t++) {
        const voxelIndex = x + y * nx + z * nx * ny + t * volumeSize;
        voxelValues.push(niftiImageBuffer[voxelIndex]);
    }

    return voxelValues;
}

// Update voxel coordinates display on the page
function updateVoxelCoordinatesDisplay(x, y, z) {
    document.getElementById('voxel-coordinates').innerText = `(x, y, z): (${x}, ${y}, ${z})`;
}

// Update the slice index based on the current view
function updateSliceIndex(view, newIndex) {
    if (view === 'axial') {
        voxel3dUnits[2] = newIndex;
    } else if (view === 'sagittal') {
        voxel3dUnits[0] = newIndex;
    } else if (view === 'coronal') {
        voxel3dUnits[1] = newIndex;
    }
    handleVoxelClick(voxel3dUnits);
}

// Load NIfTI file and display the axial, sagittal, and coronal views
function loadAllFileViews(file) {
    const fileURL = URL.createObjectURL(file);
    const imageId = `nifti:${fileURL}`;

    cornerstoneNIFTIImageLoader.nifti.loadHeader(imageId).then((header) => {
        dims = [...header.voxelLength, header.timeSlices];
        loadAndViewImage(document.getElementById('nifti-image-z'), `${imageId}#z,t-0`, 'axial');
        loadAndViewImage(document.getElementById('nifti-image-x'), `${imageId}#x,t-0`, 'sagittal');
        loadAndViewImage(document.getElementById('nifti-image-y'), `${imageId}#y,t-0`, 'coronal');
    });

}


// Plot voxel data using Plotly
function plotVoxelData(values) {
    const trace = {
        y: values,
        type: 'line',
    };
    const layout = {
        title: 'Voxel Intensity Under the Cursor',
        xaxis: { title: 'Time Point' },
        yaxis: { title: 'Intensity' },
    };
    Plotly.newPlot('plot', [trace], layout);
}

// Initialize file upload and view
document.getElementById('upload-and-view').addEventListener('click', () => {
    const file = document.getElementById('nifti-file').files[0];
    if (file) {
        getNiftiArrayBuffer(file);
        loadAllFileViews(file);
    } else {
        alert("Please select a NIFTI file to upload.");
    }
});

// Enable cornerstone for the viewports
cornerstone.enable(document.getElementById('nifti-image-z'));
cornerstone.enable(document.getElementById('nifti-image-x'));
cornerstone.enable(document.getElementById('nifti-image-y'));

// Fetch NIfTI file data as ArrayBuffer
async function getNiftiArrayBuffer(file) {
    const data = await loadNiftiFile(file);
    if (!data) return console.error('Failed to load NIfTI file');

    try {
        const header = niftiReader.readHeader(data);
        niftiImageBuffer = createTypedArray(header, niftiReader.readImage(header, data));
    } catch (error) {
        console.error('Error processing file:', error);
    }
}

// Create typed array based on NIfTI data type
// Create a mapping between datatype codes and typed array constructors
const typedArrayConstructorMap = {
    [niftiReader.NIFTI1.TYPE_UINT8]: Uint8Array,
    [niftiReader.NIFTI1.TYPE_UINT16]: Uint16Array,
    [niftiReader.NIFTI1.TYPE_UINT32]: Uint32Array,
    [niftiReader.NIFTI1.TYPE_INT8]: Int8Array,
    [niftiReader.NIFTI1.TYPE_INT16]: Int16Array,
    [niftiReader.NIFTI1.TYPE_INT32]: Int32Array,
    [niftiReader.NIFTI1.TYPE_FLOAT32]: Float32Array,
    [niftiReader.NIFTI1.TYPE_FLOAT64]: Float64Array,
    [niftiReader.NIFTI1.TYPE_RGB]: Uint8Array,
    [niftiReader.NIFTI1.TYPE_RGBA]: Uint8Array
};

// Create typed array based on NIfTI data type code
function createTypedArray(header, imageBuffer) {
    const TypedArrayConstructor = typedArrayConstructorMap[header.datatypeCode];
    
    if (TypedArrayConstructor) {
        return new TypedArrayConstructor(imageBuffer);
    } else {
        console.error('Unsupported datatype:', header.datatypeCode);
        return null;
    }
}

// Load the NIfTI file as an ArrayBuffer
async function loadNiftiFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const arrayBuffer = event.target.result;
            const data = niftiReader.isCompressed(arrayBuffer)
                ? niftiReader.decompress(arrayBuffer)
                : arrayBuffer;
            resolve(data);
        };
        reader.onerror = (error) => reject(error);
        reader.readAsArrayBuffer(file);
    });
}
