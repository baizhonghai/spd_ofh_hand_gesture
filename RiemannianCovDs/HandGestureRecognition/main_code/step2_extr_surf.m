% Cambridge_color_9_keyframe 
clc; clear all; warning off; tic
Options.upright=true;
Options.tresh=0.000001;%0.000005;%0.00001;%0.00005;%0.0001;

feature =[];
addpath(genpath('OpenSURF_version1c')) 

% 这两个目录是Cambridge的特征提取
imgDir= '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture_keyframe';
feaDir = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture_keyframe/surf_feature';

%这两个目录是North的特征提取
imgDir= '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG_keyframe';
feaDir = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG_keyframe/surf_feature5';



subdir =  dir( imgDir );
for i = 3: length( subdir )    
    subdirpath = fullfile( imgDir, subdir( i ).name);   
    subsubdirpath = dir( subdirpath ); 
    for j = 3 : length( subsubdirpath )
        subsubsubdirpath = fullfile( imgDir, subdir( i ).name, subsubdirpath( j ).name);
        images = dir( subsubsubdirpath );

        for k = 3 : length( images )
            imagepath = fullfile( imgDir, subdir( i ).name, subsubdirpath( j ).name, images( k ).name  )
            testpath = fullfile(feaDir, subdir(i).name, subsubdirpath(j).name, [images(k).name(1:10), '.mat']);
            if exist(testpath, 'file') == 2
                continue;
            end
            % 提取SURF特征
            %imagepath = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG_keyframe/08/08_Index_12/frame_0028.jpg'
            iamge = imread(imagepath) ;
            Ipts=OpenSurf(iamge,Options);
            for q = 1:size(Ipts,2)
                feature = [feature; Ipts(q).descriptor'];
            end     
            savepath = fullfile(feaDir, subdir( i ).name, subsubdirpath( j ).name, images( k ).name(1:10) );   %这个（1:10）很关键
            if ~isdir(savepath),
                mkdir(savepath);
            end;
            save(savepath, 'feature');
            rmdir(savepath)
            feature =[];
        end
    end
end
toc