// FIJI script intended to extract annotation from 'expertised' images

extension = ".tif";
dir1="/images/folder/";
dir2="/output/txt/folder/";

setBatchMode(true);
processFolder(dir1);

function processFolder(dir1) {
    list = getFileList(dir1);
    for (i=0; i<list.length; i++) {
        if (endsWith(list[i], "/"))
            processFolder(dir1+list[i]);
        else if (endsWith(list[i], extension))
        processImage(dir1, list[i]);
    }
}

function processImage(dir1, name) {
    open(dir1+name);
    if (selectionType()!=-1){
        roiManager("Add");
        roiManager("Measure");
        saveAs("Results", dir2+name+".txt");
        run("Clear Results");
        selectWindow("Results");
        run("Close");
        roiManager("Delete");
        close();
    }
}
