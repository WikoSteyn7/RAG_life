import json
from typing import Any, AsyncGenerator

import openai
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType

from approaches.approach import Approach
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_token_limit
from text import nonewlines


class ChatReadRetrieveReadApproach(Approach):
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    NO_RESPONSE = "0"

    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
    
    selection = ["Discovery", "Onespark"]
#     combined_prompt = f"""Do the following step by step. 
# Step 1:You are an assistant tasked with retrieving detailed information regarding life insuarance policies from the provided sources below. The Sources for {selection[0]} is given below <<< {selection[0]} Sources:>>> and for {selection[1]} below <<< {selection[1]} Sources:>>> 
# Start by reading through all the sources. It is imperative to keep {selection[0]} and {selection[1]} information seperate.
# Step 2: You are an assistant that helps people compare life insuarance policies. For each query always give correct answer for {selection[0]} and {selection[1]}. 
# You are not allowed to mix {selection[0]} and {selection[1]} Sources when answering the query.   
# The fromat you should always follow is shown below:
# For comparitive queries use bulletpoints if applicable and for procedural queries give the answer as steps. You should always give the answer for {selection[0]}, {selection[1]} in this format.
# If there is no relevant source for a policy, say you don't know.
# Be verbose and don't leave out anything of importance. 
# It is imperative to maintain consistency, meaning that if the same query is made in the future, your response should remain the same.
# Always start with a friendly greeting like: Hi, it is Theo here and these are the results I found, let me kmow if you need anything else.
# When presenting tabular information, format it as an HTML table, not in markdown.
# Please refrain from generating answers that do not rely on the sources below. Instead, answer only with facts found within the list of sources provided.
# Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf] 
# You are not allowed to give a statement without a source and only use the above format for any source.
# """

    combined_prompt =f"""
Please follow these steps to ensure accurate and consistent information:
Step 1:
- **Objective**: Your task is to retrieve information regarding life insurance policies from the provided sources.
- **Sources**:
  - For {selection[0]}: Refer to the section below <<< {selection[0]} Sources:>>>.
  - For {selection[1]}: Refer to the section below <<< {selection[1]} Sources:>>>.
- **Instructions**: Thoroughly read through all the available sources, ensuring that information from {selection[0]} and {selection[1]} remains distinct and separate.
Step 2:
- **Objective**: 
  - You are a life insuarance policy expert that helps consultants to compare policies by using the information gathered in Step 1. 
  - Respond accurately to each query for both {selection[0]} and {selection[1]}, but never mix {selection[0]} and {selection[1]} Sources. 
  - **Format**:
  - For comparative queries: Utilize bullet points if applicable. 
  - For procedural queries: Present the information in a step-by-step format.
  - Split the answer into a html table with two columns one for {selection[0]} and one for {selection[1]}
  - If information for a policy is not available in the sources, clearly state that you do not have the necessary information.
  - When presenting tabular information, format it as an HTML table, not in markdown.
  - If a source does not include relevant information, do not mention it.
  - A friendly greeting such as, "Hi, it's Theo here. I have found the following results, let me know if you need anything else," should preface your responses.
  - Always answer in the language used by the user in the query.
- **Details**: Be verbose in your responses, but only give information that is relevant to the user query.  
- **Consistency**: Ensure that your responses remain consistent for repeated queries.
- **Sourcing**: 
  -Each source has a name followed by colon and the actual information, only include the source name at the end of each fact . Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf] 
  -Do not mention filenames in any other way.
  -You are not allowed to give a statement without a source and only use the above format for all sourcing. 

Remember, accuracy, clarity, and Consistency are key! Feel free to ask if you have any questions that might help you produce a better answer.
"""

    system_message_chat_conversation = """Always start with a friendly greeting like: Hi, it is Theo here and these are the results I found, let me kmow if you need anything else. 
You are an assistant that helps people compare funeral policies. Your sources below consists of a Capitec and Standard Bank policies. 
For each query give the correct response from Capitec and Standard Bank seperately. For comparitive queries use bulletpoints if applicable and for procedural queries give the answer as steps.
Be verbose and don't leave out anything of importance.  
Be deterministic, it is of vital importance to give the same answer in the future if the query is the same.
Answer ONLY with the facts listed in the list of sources below.  If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, always answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Always use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf] 
You are not allowed to give a statement without a source and only use the above format for any source.
"""
    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about their funeral policy.
Use double angle brackets to reference the questions, e.g. <<How do I cancel my policy?>>.
Try not to repeat questions that have already been asked.
Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about the two Life Insuarance policies.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Ensure that your responses remain consistent for repeated queries.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
If you cannot generate a search query, return just the number 0.
"""
    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'What benefits are included?' },
        {'role' : ASSISTANT, 'content' : f'Show all information regarding {selection[0]} and {selection[0]} benefits in a comparitive format' },
        {'role' : USER, 'content' : 'What is the waiting period?' },
        {'role' : ASSISTANT, 'content' : f'Show all information regarding {selection[0]} and {selection[1]} waiting period in a comparitive format' }
    ]

    def __init__(
        self,
        search_client: SearchClient,
        openai_host: str,
        chatgpt_deployment: str,
        chatgpt_model: str,
        embedding_deployment: str,
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_host = openai_host
        self.chatgpt_deployment = chatgpt_deployment
        self.chatgpt_model = "gpt-3.5-turbo-16k"
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run_until_final_call(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        filter = self.build_filter(overrides, auth_claims)

        user_query_request = "Generate search query for: " + history[-1]["user"]

        functions = [
            {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure Cognitive Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            }
        ]

        chatgpt_args = {"deployment_id": self.chatgpt_deployment} if self.openai_host == "azure" else {}
        
        # # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        # messages = self.get_messages_from_history(
        #     self.query_prompt_template,
        #     self.chatgpt_model,
        #     history,
        #     user_query_request,
        #     self.query_prompt_few_shots,
        #     self.chatgpt_token_limit - len(user_query_request),
        # )

        
        # chat_completion = await openai.ChatCompletion.acreate(
        #     **chatgpt_args,
        #     model=self.chatgpt_model,
        #     messages=messages,
        #     temperature=0.2,
        #     max_tokens=80,
        #     n=1,
        #     functions=functions,
        #     function_call="auto",
        # )

        # query_text = self.get_search_query(chat_completion, history[-1]["user"])
        # print(query_text) 
        query_text = history[-1]["user"]
        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        if has_vector:
            embedding_args = {"deployment_id": self.embedding_deployment} if self.openai_host == "azure" else {}
            embedding = await openai.Embedding.acreate(**embedding_args, model=self.embedding_model, input=query_text)
            query_vector = embedding["data"][0]["embedding"]
        else:
            query_vector = None

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        if overrides.get("semantic_ranker") and has_text:
            capi = await self.search_client.search(query_text,
                                          filter="sourcefile eq 'Discovery life plan.pdf'",
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                          vector=query_vector,
                                          top_k=20 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
     
          
            
            sb = await self.search_client.search(query_text,
                                filter="sourcefile eq 'Onespark life policy.pdf'",
                                query_type=QueryType.SEMANTIC,
                                query_language="en-us",
                                query_speller="lexicon",
                                semantic_configuration_name="default",
                                top=top,
                                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                                vector=query_vector,
                                top_k=20 if query_vector else None,
                                vector_fields="embedding" if query_vector else None)
        else:
            capi = await self.search_client.search(query_text,
                                          filter="sourcefile eq 'Discovery life plan.pdf'",
                                          top=top,
                                          vector=query_vector,
                                          top_k=20 if query_vector else None,
                                          vector_fields="embedding" if query_vector else None)
                        
            sb = await self.search_client.search(query_text,
                    filter="sourcefile eq 'Onespark life policy.pdf'",
                    top=top,
                    vector=query_vector,
                    top_k=20 if query_vector else None,
                    vector_fields="embedding" if query_vector else None) 
            

            
        if use_semantic_captions:
            resultsCapi = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in capi]
        else:
            resultsCapi = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in capi]        
        contentCapi = "\n".join(resultsCapi)
        
        ####
        if use_semantic_captions:
            resultsSB = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) async for doc in sb]
        else:
            resultsSB = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) async for doc in sb]
        contentSB = "\n".join(resultsSB)
        ####
        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""


        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            system_message = self.combined_prompt.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif prompt_override.startswith(">>>"):
            system_message = self.combined_prompt.format(
                injected_prompt=prompt_override[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            system_message = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt)

     ####
        policy_message =self.combined_prompt
        
        messagesQA = self.get_messages_from_history(
            policy_message,
            self.chatgpt_model,
            history,
            history[-1]["user"]+ "\n\n" + f"<<< {self.selection[0]} Sources:>>>\n" + contentCapi + f"<<< {self.selection[1]} Sources:>>>\n" + contentSB , # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            max_tokens=12000)
        msg_to_display = '\n\n'.join([str(message) for message in messagesQA])
        
        ####
        extra_info = {"data_points": resultsCapi + resultsSB, "thoughts": f"Searched for:<br>{query_text}<br><br>Conversations:<br>" + msg_to_display.replace('\n', '<br>')}
              

        qaResult = openai.ChatCompletion.acreate(
            **chatgpt_args,
            model=self.chatgpt_model,
            messages=messagesQA,
            temperature=overrides.get("temperature") or 0.0,
            max_tokens=4096,
            n=1,
            stream=should_stream)
        
        return (extra_info, qaResult)

    async def run_without_streaming(
        self, history: list[dict[str, str]], overrides: dict[str, Any], auth_claims: dict[str, Any]) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(history, overrides, auth_claims, should_stream=False)
        chat_resp = await chat_coroutine
        chat_content = chat_resp.choices[0].message.content
        extra_info["answer"] = chat_content
        return extra_info

    async def run_with_streaming(
        self, history: list[dict[str, str]], overrides: dict[str, Any], auth_claims: dict[str, Any]) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(history, overrides, auth_claims, should_stream=True)
        yield extra_info
        async for event in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            if event["choices"]:
                yield event

    def get_messages_from_history(
        self,
        system_prompt: str,
        model_id: str,
        history: list[dict[str, str]],
        user_conv: str,
        few_shots=[],
        max_tokens: int = 8192,
    ) -> list:
        message_builder = MessageBuilder(system_prompt, model_id)

        # Add examples to show the chat what responses we want. It will try to mimic any responses and make sure they match the rules laid out in the system message.
        for shot in few_shots:
            message_builder.append_message(shot.get("role"), shot.get("content"))

        user_content = user_conv
        append_index = len(few_shots) + 1

        message_builder.append_message(self.USER, user_content, index=append_index)

        for h in reversed(history[:-1]):
            if bot_msg := h.get("bot"):
                message_builder.append_message(self.ASSISTANT, bot_msg, index=append_index)
            if user_msg := h.get("user"):
                message_builder.append_message(self.USER, user_msg, index=append_index)
            if message_builder.token_length > max_tokens:
                break

        messages = message_builder.messages
        return messages

    def get_search_query(self, chat_completion: dict[str, any], user_query: str):
        response_message = chat_completion["choices"][0]["message"]
        if function_call := response_message.get("function_call"):
            if function_call["name"] == "search_sources":
                arg = json.loads(function_call["arguments"])
                search_query = arg.get("search_query", self.NO_RESPONSE)
                if search_query != self.NO_RESPONSE:
                    return search_query
        elif query_text := response_message.get("content"):
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query
