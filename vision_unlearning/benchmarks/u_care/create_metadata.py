import os
import json

# Global variables

theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
"Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
"Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
"Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
"Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
"Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
"Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
"Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]

class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
"Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"]


def get_forget_retain_metadata(images_path, metadata_path, forget_class=None):
    global theme_available, class_available
    # iterate over the style folders to get the classes
    # create temp folders with the forget and retain set

    overwrite_class = None

    # get index of forget class
    if forget_class is not None:
        try:
            forget_class_index = class_available.index(forget_class)
        except:
            print("Forget class not found!")
            return -1, overwrite_class

    try:
        os.makedirs(metadata_path, exist_ok=True)
    except:
        print("Could not create folder, check if temp folder are still there!")
        return -1, overwrite_class

    # select next class as overwrite
    if forget_class_index == len(class_available) - 1:
        overwrite_class = class_available[0]
    else:
        overwrite_class = class_available[forget_class_index + 1]

    # create metadata file for forget class
    metadata_forget = []
    metadata_retain = []

    images = os.listdir(images_path)

    prompts_forget, prompts_retain, prompts_overwrite = [], [], []

    # iterate over style folders and copy files to the desired locations
    for theme in theme_available:
        
        for class_name in class_available:

            filter_name = f"{class_name}-{theme}-"
            selected_images = [img for img in images if img.startswith(filter_name)]

            if class_name == forget_class:
                for image in selected_images:
                    try:
                        # create corresponding prompt for image
                        if theme == "Seed_Images":
                            prompt_forget = f"An image of {class_name} in Photo style."
                            prompt_overwrite = f"An image of {overwrite_class} in Photo style."
                        else:
                            prompt_forget = f"An image of {class_name} in {theme.replace('_', ' ')} style."
                            prompt_overwrite = f"An image of {overwrite_class} in {theme.replace('_', ' ')} style."

                        metadata_forget.append({
                            "file_name": os.path.join(images_path, image),
                            "text": prompt_forget,
                            "overwrite": prompt_overwrite
                        })

                    except Exception as e:
                        # Catches any other unexpected errors
                        print(f"An unexpected error occurred: {e}")
                        print(f"Could not create metadata info for {image} in forget set!")

            else:
                # copy image to retain set folder
                for image in selected_images:
                    try:
                        # create corresponding prompt for image
                        if theme == "Seed_Images":
                            prompt_retain = f"An image of {class_name} in Photo style."
                        else:
                            prompt_retain = f"An image of {class_name} in {theme.replace('_', ' ')} style."

                        metadata_retain.append({
                            "file_name": os.path.join(images_path, image),
                            "text": prompt_retain,
                            "overwrite": ''
                        })
                    except Exception as e:
                        # Catches any other unexpected errors
                        print(f"An unexpected error occurred: {e}")
                        print(f"Could not create metadata info for {image} in retain set!")

    final_forget_data = {
        "train": metadata_forget
    }
    forget_filename = f"metadata-{forget_class}-forget.json" # Change extension to .json
    with open(os.path.join(metadata_path, forget_filename), 'w') as f:
        # Use json.dump to write the entire structure (list under "train")
        json.dump(final_forget_data, f, indent=4)

    final_retain_data = {
        "train": metadata_retain
    }
    retain_filename = f"metadata-{forget_class}-retain.json" # Change extension to .json
    with open(os.path.join(metadata_path, retain_filename), 'w') as f:
        # Use json.dump to write the entire structure (list under "train")
        json.dump(final_retain_data, f, indent=4)

    return 0, overwrite_class


def get_forget_retain_metadata_theme(images_path, metadata_path, forget_theme=None, overwrite_theme=None):
    global theme_available, class_available
    # iterate over the style folders to get the classes
    # create temp folders with the forget and retain set
    
    # get index of forget class
    if forget_theme is not None:
        try:
            forget_theme_index = theme_available.index(forget_theme)
        except:
            print("Forget theme not found!")
            return -1, overwrite_theme

    try:
        os.makedirs(metadata_path, exist_ok=True)
    except:
        print("Could not create folder, check if temp folder are still there!")
        return -1, overwrite_theme

    # select next class as overwrite
    if overwrite_theme is not None:
        try:
            overwrite_theme_index = theme_available.index(overwrite_theme)
        except:
            print("Overwrite theme not found!")
            return -1, overwrite_theme
    else:
        if forget_theme_index == len(theme_available) - 1:
            overwrite_theme = theme_available[0]
        else:
            overwrite_theme = theme_available[forget_theme_index + 1]

    if overwrite_theme == "Seed_Images":
        overwrite_theme = "Photo"

    # create metadata file for forget class
    metadata_forget = []
    metadata_retain = []

    images = os.listdir(images_path)

    prompts_forget, prompts_retain, prompts_overwrite = [], [], []

    # iterate over style folders and copy files to the desired locations
    for theme in theme_available:
        
        for class_name in class_available:

            filter_name = f"{class_name}-{theme}-"
            selected_images = [img for img in images if img.startswith(filter_name)]

            if theme == forget_theme:
                for image in selected_images:
                    try:
                        # create corresponding prompt for image
                        if theme == "Seed_Images":
                            prompt_forget = f"An image of {class_name} in Photo style."
                        else:
                            prompt_forget = f"An image of {class_name} in {theme.replace('_', ' ')} style."

                        prompt_overwrite = f"An image of {class_name} in {overwrite_theme.replace('_', ' ')} style."

                        metadata_forget.append({
                            "file_name": os.path.join(images_path, image),
                            "text": prompt_forget,
                            "overwrite": prompt_overwrite
                        })

                    except Exception as e:
                        # Catches any other unexpected errors
                        print(f"An unexpected error occurred: {e}")
                        print(f"Could not create metadata info for {image} in forget set!")

            else:
                # copy image to retain set folder
                for image in selected_images:
                    try:
                        # create corresponding prompt for image
                        if theme == "Seed_Images":
                            prompt_retain = f"An image of {class_name} in Photo style."
                        else:
                            prompt_retain = f"An image of {class_name} in {theme.replace('_', ' ')} style."

                        metadata_retain.append({
                            "file_name": os.path.join(images_path, image),
                            "text": prompt_retain,
                            "overwrite": ''
                        })
                    except Exception as e:
                        # Catches any other unexpected errors
                        print(f"An unexpected error occurred: {e}")
                        print(f"Could not create metadata info for {image} in retain set!")

    final_forget_data = {
        "train": metadata_forget
    }
    forget_filename = f"metadata-{forget_theme}-forget.json" # Change extension to .json
    with open(os.path.join(metadata_path, forget_filename), 'w') as f:
        # Use json.dump to write the entire structure (list under "train")
        json.dump(final_forget_data, f, indent=4)

    final_retain_data = {
        "train": metadata_retain
    }
    retain_filename = f"metadata-{forget_theme}-retain.json" # Change extension to .json
    with open(os.path.join(metadata_path, retain_filename), 'w') as f:
        # Use json.dump to write the entire structure (list under "train")
        json.dump(final_retain_data, f, indent=4)

    return 0, overwrite_theme


if __name__ == "__main__":
    images_path = "assets/data/images"
    metadata_path = "assets/data/forget_classes"

    for class_name in class_available:
        print(f"Creating metadata for forget class: {class_name}")
        status, overwrite_class = get_forget_retain_metadata(images_path, metadata_path, forget_class=class_name)
        if status == 0:
            print(f"Metadata created for forget class: {class_name}, overwrite class: {overwrite_class}")
        else:
            print(f"Failed to create metadata for forget class: {class_name}")
    
    metadata_path = "assets/data/forget_styles"

    for theme_name in theme_available:
        print(f"Creating metadata for forget style: {theme_name}")
        status, overwrite_theme = get_forget_retain_metadata_theme(images_path, metadata_path, forget_theme=theme_name, overwrite_theme='Seed_Images')
        if status == 0:
            print(f"Metadata created for forget style: {theme_name}, overwrite class: {overwrite_theme}")
        else:
            print(f"Failed to create metadata for forget class: {theme_name}")