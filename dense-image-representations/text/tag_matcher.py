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


class TagMatcher:
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
        self,
        tags_lists_1: List[List[str]],
        tags_lists_2: List[List[str]],
        temperature=0.6,
        top_p=0.9,
        max_gen_len=None,
    ) -> List[str]:
        """
        Use Llama 2 to match tags in two list.

        Args:
            tags_1 (List[str]): The first list of tags to be matched.
            tags_2 (List[str]): The second list of tags to be matched.
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
                    Your job is to match objects in one list to the closest synonym
                    in another list. Do not output a match if the second list doesn't have a matching item.
                    Output a single JSON object with the objects in the first list as keys and the matching
                    objects in the second list as values.

                    Example:

                    Input:
                    
                        List 1: [a small toilet stall, a toilet brush, three rolls, toilet paper]
                        List 2: [wall-tile, paper-merged, toilet, door, floor-other-merged]

                    Output:
                        {
                            "a small toilet stall": "toilet",
                            "toilet paper": "paper-merged"
                        }
                    
                    """,
                },
                {
                    "role": "user",
                    "content": """List 1: [the leaves, people]
                    List2: [person, grass, tree, sky, person]""",
                },
                {
                    "role": "assistant",
                    "content": """
                        {
                            "the leaves": "tree",
                            "people": "person"
                        }
                """,
                },
                {
                    "role": "user",
                    "content": f"""Match the following lists:
                List 1: {tags_1}
                List 2: {tags_2}
            and output just a single JSON object in the same format as above without any explanation.""",
                },
            ]
            for tags_1, tags_2 in zip(tags_lists_1, tags_lists_2)
        )

        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        results = [result["generation"]["content"] for result in results]
        return results

    def parse(
        self, tags_lists_1: List[List[str]], tags_lists_2: List[List[str]]
    ) -> List[dict[str, str]]:
        """
        Parse the captions to extract objects, attributes and the relationships between them.
        Note that this function accepts multiple pairs of lists.

        Args:
            tags_lists_1 (List[List[str]]): The list of first list of tags to be matched.
            tags_lists_2 (List[List[str]]): The list of second list of tags to be matched.

        Returns:
            List[dict[str, str]]: The parsed captions in the following format:
                {
                    "idx of object in tags_lists_1": "idx of object in tags_list_2",
                }
        """

        results = self.ask_llama(tags_lists_1, tags_lists_2)
        outputs = [{}] * len(results)
        for i, result in enumerate(results):
            try:
                decoded_result = json.loads(result)
            except json.decoder.JSONDecodeError as e:
                decoded_result = {}
            finally:
                for object_1, object_2 in decoded_result.items():
                    idx_1 = tags_lists_1[i].index(object_1)
                    idx_2 = tags_lists_2[i].index(object_2)
                    outputs[i][idx_1] = idx_2
        return outputs


def main(_):
    matcher = TagMatcher()


if __name__ == "__main__":
    app.run(main)
