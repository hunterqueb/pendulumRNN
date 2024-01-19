from moviepy.editor import VideoFileClip
import os

def convert_gif_to_mp4(gif_filename, output_dir):
    # Extract the base name of the file (without path)
    base_name = os.path.basename(gif_filename)

    # Create the output filename by replacing the .gif extension with .mp4
    # and saving it in the output directory
    output_filename = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.mp4')

    # Check if the MP4 file already exists
    if os.path.exists(output_filename):
        print(f"Skipping conversion, MP4 already exists for {gif_filename}")
        return

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the GIF using MoviePy
    clip = VideoFileClip(gif_filename)

    # Write the clip to a file in MP4 format
    clip.write_videofile(output_filename, codec="libx264")

    print(f"Converted {gif_filename} to {output_filename}")

def find_and_convert_gifs(directory):
    # Define the output directory for the converted videos
    output_dir = os.path.join(directory, "videos")

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Search for GIF files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.gif'):
            # Full path of the GIF file
            full_path = os.path.join(directory, filename)
            convert_gif_to_mp4(full_path, output_dir)

def main():
    # Run the conversion on the 'predict' directory
    predict_directory = 'predict' # Path to the 'predict' directory
    find_and_convert_gifs(predict_directory)

if __name__ == "__main__":
    main()
