function create_video_from_images(num_images, common_string)
% This function creates a video file (.avi) from a sequence of PNG images.
% All the images provided should be sequentially numbered and saved in the
% local folder.

% Inputs:
% num_images: a 1 x 1 scalar that provides the number of images used.
% common_string: a string that is common to all the numbered images.

% Outputs:
% A video file is saved in the local folder with the name
% common_stringvideo.avi

% Reference: https://www.mathworks.com/matlabcentral/answers/153925-how-to-make-a-video-from-images

     % Load the images
     images    = cell(num_images, 1);
     for i = 1:length(images)
         images{i} = imread(strcat(common_string, num2str(i), '.png'));
     end
    
     % Create a video writer object:
     writerObj = VideoWriter(strcat(common_string, 'video.avi')); 
    
     % Open the video writer:
     open(writerObj);
    
     % Write the frames to the video:
     for u=1:length(images)
    
         % Convert each image to a frame:
         frame = im2frame(images{u});
    
         % Write each frame to the video writer object:
          writeVideo(writerObj, frame);
       
     end
    
     % Close the writer object:
     close(writerObj);

end