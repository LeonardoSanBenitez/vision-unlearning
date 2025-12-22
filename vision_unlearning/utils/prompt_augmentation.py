import copy
from enum import Enum
import regex as re
import openai as OpenAI

BASE_URL = ''
API_KEY = ''


class ConceptType(str, Enum):
    """Enum representing the type of concept to unlearn."""
    Object = "object"
    Art = "art"


def uce_prompt_augmentation(expand_prompts: bool,edit_list: list[str],guide_list: list[str],concept_type: ConceptType):
    if expand_prompts:
        edit_copy = copy.deepcopy(edit_list)
        guide_copy = copy.deepcopy(guide_list)

        for concept, guide_concept in zip(edit_copy, guide_copy):
            if concept_type == ConceptType.Art:
                edit_list.extend([
                    f"painting by {concept}", f"art by {concept}",
                    f"artwork by {concept}", f"picture by {concept}",
                    f"style of {concept}"
                ])
                guide_list.extend([
                    f"painting by {guide_concept}", f"art by {guide_concept}",
                    f"artwork by {guide_concept}", f"picture by {guide_concept}",
                    f"style of {guide_concept}"
                ])
            else:
                edit_list.extend([
                    f"image of {concept}", f"photo of {concept}",
                    f"portrait of {concept}", f"picture of {concept}",
                    f"painting of {concept}", f"picture of {concept} doing something"
                ])
                guide_list.extend([
                    f"image of {guide_concept}", f"photo of {guide_concept}",
                    f"portrait of {guide_concept}", f"picture of {guide_concept}",
                    f"painting of {guide_concept}", f"picture of {concept} doing something"
                ])

    return edit_list, guide_list


def mace_prompt_augmentation(content, augment=True, sampled_indices=None, concept_type='object'):
    if augment:
        # some sample prompts provided
        if concept_type == 'object':
            prompts = [
                # object augmentation
                ("{} in a photo".format(content), content),
                ("{} in a snapshot".format(content), content),
                ("A snapshot of {}".format(content), content),
                ("A photograph showcasing {}".format(content), content),
                ("An illustration of {}".format(content), content),
                ("A digital rendering of {}".format(content), content),
                ("A visual representation of {}".format(content), content),
                ("A graphic of {}".format(content), content),
                ("A shot of {}".format(content), content),
                ("A photo of {}".format(content), content),
                ("A black and white image of {}".format(content), content),
                ("A depiction in portrait form of {}".format(content), content),
                ("A scene depicting {} during a public gathering".format(content), content),
                ("{} captured in an image".format(content), content),
                ("A depiction created with oil paints capturing {}".format(content), content),
                ("An image of {}".format(content), content),
                ("A drawing capturing the essence of {}".format(content), content),
                ("An official photograph featuring {}".format(content), content),
                ("A detailed sketch of {}".format(content), content),
                ("{} during sunset/sunrise".format(content), content),
                ("{} in a detailed portrait".format(content), content),
                ("An official photo of {}".format(content), content),
                ("Historic photo of {}".format(content), content),
                ("Detailed portrait of {}".format(content), content),
                ("A painting of {}".format(content), content),
                ("HD picture of {}".format(content), content),
                ("Magazine cover capturing {}".format(content), content),
                ("Painting-like image of {}".format(content), content),
                ("Hand-drawn art of {}".format(content), content),
                ("An oil portrait of {}".format(content), content),
                ("{} in a sketch painting".format(content), content),
            ]
      
        elif concept_type == 'style':
            # art augmentation
            prompts = [
                ("An artwork by {}".format(content), content),
                ("Art piece by {}".format(content), content),
                ("A recent creation by {}".format(content), content),
                ("{}'s renowned art".format(content), content),
                ("Latest masterpiece by {}".format(content), content),
                ("A stunning image by {}".format(content), content),
                ("An art in {}'s style".format(content), content),
                ("Exhibition artwork of {}".format(content), content),
                ("Art display by {}".format(content), content),
                ("a beautiful painting by {}".format(content), content),
                ("An image inspired by {}'s style".format(content), content),
                ("A sketch by {}".format(content), content),
                ("Art piece representing {}".format(content), content),
                ("A drawing by {}".format(content), content),
                ("Artistry showcasing {}".format(content), content),
                ("An illustration by {}".format(content), content),
                ("A digital art by {}".format(content), content),
                ("A visual art by {}".format(content), content),
                ("A reproduction inspired by {}'s colorful, expressive style".format(content), content),
                ("Famous painting of {}".format(content), content),
                ("A famous art by {}".format(content), content),
                ("Artistic style of {}".format(content), content),
                ("{}'s famous piece".format(content), content),
                ("Abstract work of {}".format(content), content),
                ("{}'s famous drawing".format(content), content),
                ("Art from {}'s early period".format(content), content),
                ("A portrait by {}".format(content), content),
                ("An imitation reflecting the style of {}".format(content), content),
                ("An painting from {}'s collection".format(content), content),
                ("Vibrant reproduction of artwork by {}".format(content), content),
                ("Artistic image influenced by {}".format(content), content),
            ]
        else:
            raise ValueError("unknown concept type.")
    else:
        prompts = [
            ("A photo of {}".format(content), content),
        ]

    if sampled_indices is not None:
        sampled_prompts = [prompts[i] for i in sampled_indices if i < len(prompts)]
    else:
        sampled_prompts = prompts
    
    return sampled_prompts

def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(
        r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(
        r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', '') for x in class_prompt_collection]
    return class_prompt_collection


def text_augmentation(erased_concept, mapping_concept, concept_type, num_text_augmentations=100):

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    class_prompt_collection = []

    if concept_type == 'object':
        messages = [
            {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
            {"role": "user", "content": f"Generate {num_text_augmentations} captions for images containing {erased_concept}. The caption should also contain the word '{erased_concept}'. Please do not use any emojis in the captions."},
        ]
  
        while True:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            class_prompt_collection += [x for x in completion.choices[0].message.content.lower(
            ).split('\n') if erased_concept in x]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content})
            messages.append(
                {"role": "user", "content": f"Generate {num_text_augmentations-len(class_prompt_collection)} more captions"})
            if len(class_prompt_collection) >= num_text_augmentations:
                break
  
        class_prompt_collection = clean_prompt(class_prompt_collection)[:num_text_augmentations]
        class_prompt_formated = []
        mapping_prompt_formated = []
        
        for prompt in class_prompt_collection:
            class_prompt_formated.append((prompt, erased_concept))
            mapping_prompt_formated.append((prompt.replace(erased_concept, mapping_concept), mapping_concept))

        return class_prompt_formated, mapping_prompt_formated
