from typing import List, Optional

from absl import flags, app
import json
import os
import pdb

from llama import Llama, Dialog

from datasets import load_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ckpt_dir",
    "/cmlscratch/pkattaki/developer/llama/llama-2-7b-chat/",
    "Path to the Llama 2 chat model checkpoint directory.",
)
flags.DEFINE_string(
    "tokenizer_path",
    "/cmlscratch/pkattaki/developer/llama/tokenizer.model",
    "Path to the Llama 2 chat model tokenizer.",
)


class CaptionParser:
    def __init__(
        self,
        max_seq_len: int = 1024,
        max_batch_size: int = 8,
    ):
        self.generator = Llama.build(
            ckpt_dir=FLAGS.ckpt_dir,
            tokenizer_path=FLAGS.tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def ask_llama(
        self, captions: List[str], temperature=0.6, top_p=0.9, max_gen_len=None
    ) -> List[str]:
        """
        Use Llama 2 to parse the captions.

        Args:
            captions (List[str]): The input captions to be parsed.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
        """
        dialogs: List[Dialog] = list(
            [
                {
                    "role": "system",
                    "content": """You are a helpful assistant who is designed to parse sentences.
                Given a sentence, your first task is to identify all the nouns and noun phrases in the sentence. These are the objects.
                The second task is to qualify all objects with the corresponding attributes available in the sentence. These are generally adjectives that describe them.
                The final task is to extract relationships in the sentence. These are generally verbs and prepositions in the sentence that describe the action or interaction between the identified objects.
                """,
                },
                {
                    "role": "user",
                    "content": f"""In the following sentence:
                    a tennis player in all white swinging at the ball
                list the objects (nouns) with their attributes (adjectives) and relationships (verbs, prepositions) between them.
                Output should be in JSON format with fields 'objects', 'relationships'
                Each relationship is a list of the form [subject, object, predicate]
                such that the subject is performing the action described by the predicate on the object.""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                        "objects":["tennis player in all white", "ball"],
                        "relationships":[["tennis player", "ball", "swinging at"]]
                    }
                """,
                },
                {
                    "role": "user",
                    "content": f"""A young man wearing black attire and a flowered tie is standing and smiling.""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                        "objects":["young man", "black attire", "flowered tie"],
                        "relationships":[["young man", "black attire", "wearing"], ["young man", "flowered tie", "wearing"]]
                    }
                """,
                },
                {
                    "role": "user",
                    "content": f"""In the following sentence:
                    three people sit at a table holding lollipops
                The word people is qualified with the number 'three'. Therefore, this should be split into three separate objects.""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                        "objects":["person-1", "person-2", "person-3", "lollipops", "table"],
                        "relationships":[["person-1", "table", "sit"], ["person-2", "table", "sit"], ["person-3", "table", "sit"], 
                        ["person-1", "lollipop", "holding"], ["person-2", "lollipop", "holding"], ["person-3", "lollipop", "holding"]]
                    }
                """,
                },
                {
                    "role": "user",
                    "content": f"""the person without earrings pays the person with earrings""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                        "objects":["person without earrings", "person with earrings"],
                        "relationships":[["person without earrings", "person with earrings", "pays"]]
                    }
                """,
                },
                {
                    "role": "user",
                    "content": f"""Parse the following sentence:
                    {caption}
                and output just a single JSON object in the same format as above without any explanation.""",
                },
            ]
            for caption in captions
        )

        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        results = [result["generation"]["content"] for result in results]
        return results

    def parse(self, captions: List[str]) -> List[dict[str, str]]:
        """
        Parse the captions to extract objects, attributes and the relationships between them.

        Args:
            captions (List[str]): The input captions to be parsed.

        Returns:
            List[dict[str, str]]: The parsed captions in the following format:
                {
                    "objects": List[str],
                    "relationships": List[List[str, str, str]]
                }
        """

        def get_object_index(key: str, objects: List[str]) -> Optional[int]:
            for i, obj in enumerate(objects):
                if key in obj:
                    return i
            return None

        results = self.ask_llama(captions)
        outputs = []
        for result in results:
            try:
                decoded_result = json.loads(result)
            except json.decoder.JSONDecodeError as e:
                decoded_result = {"objects": [], "relationships": []}
            finally:
                indexed_relationships = []
                for relationship in decoded_result["relationships"]:
                    if len(relationship) == 3 and None not in relationship:
                        indexed_relationship = [
                            get_object_index(relationship[0], decoded_result["objects"]),
                            get_object_index(relationship[1], decoded_result["objects"]),
                            relationship[2],
                        ]
                        if None not in indexed_relationship:
                            indexed_relationships.append(indexed_relationship)
                decoded_result["relationships"] = indexed_relationships
                outputs.append(decoded_result)
        return outputs


def main(_):
    parser = CaptionParser()
    # print(
    #     parser.parse(
    #         captions=[
    #             "the dog is running on the field.",
    #             "the cat is sitting on the mat.",
    #             "three people sit at a table holding lollipops.",
    #             "a slice of pepperoni pizza on a piece of parchment paper on a paper plate.",
    #             "tennis player in all white swinging at a ball.",
    #         ],
    #     )
    # )

    # Parse COCO captions
    # f = open('/fs/cml-datasets/coco/annotations/captions_train2017.json')
    # d = json.load(f)['annotations']
    # save_path = '../coco_parsed_captions'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # for ann in d:
    #     image_id = ann['image_id']
    #     cap_id = ann['id']
    #     out = parser.parse([ann['caption']])[0]
    #     if len(out['objects']) > 0 and len(out['relationships']) > 0:
    #         with open(f'{save_path}/{image_id}_{cap_id}.json', 'w+') as f:
    #             out['caption'] = ann['caption']
    #             json.dump(out, f)

    # Parse Winoground captions
    # save_path = '../winoground_parsed_captions'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # winoground_dataset = load_dataset('facebook/winoground', use_auth_token='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', cache_dir = '/cmlscratch/nehamk/datasets')
    # data = winoground_dataset['test']
    # for i in range(10):
    #     cap_0 = data[i]['caption_0']
    #     cap_1 = data[i]['caption_1']
    #     out = parser.parse([cap_0, cap_1])
    #     with open(f'{save_path}/{i}_0.json', 'w+') as f:
    #         out[0]['caption'] = cap_0
    #         json.dump(out[0], f)
    #     with open(f'{save_path}/{i}_1.json', 'w+') as f:
    #         out[1]['caption'] = cap_1
    #         json.dump(out[1], f)

if __name__ == "__main__":
    app.run(main)
