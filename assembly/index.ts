import { collections } from "@hypermode/modus-sdk-as";
import { starterEmojis } from "./emojis";
import { models } from "@hypermode/modus-sdk-as";
import { EmbeddingsModel } from "@hypermode/modus-sdk-as/models/experimental/embeddings";

import {
  OpenAIChatModel,
  ResponseFormat,
  SystemMessage,
  UserMessage,
} from "@hypermode/modus-sdk-as/models/openai/chat";
import { JSON } from "json-as";

// These names should match the ones defined in the hypermode.json manifest file.
const miniLMEmbeddingsModelName: string = "minilm";
const generationModelName: string = "text-generator";
const emojis: string = "emojis";
const searchMethod: string = "searchMethod1";

export function miniLMEmbed(text: string[]): f32[][] {
  const model = models.getModel<EmbeddingsModel>(miniLMEmbeddingsModelName);
  const input = model.createInput(text);
  const output = model.invoke(input);
  return output.predictions;
}

export function getEmojiFromString(text: string): string {
  if (
    text.length >= 2 &&
    0xd800 <= text.charCodeAt(0) &&
    text.charCodeAt(0) <= 0xdbff
  ) {
    // The first character is a high surrogate, return the first two characters
    return text.substring(0, 2);
  } else {
    // The first character is not a high surrogate, return the first character
    return text.substring(0, 1);
  }
}

// see examples/textgeneration for explanation
function generateListForEmoji(text: string): string[] {
  // Prompt trick: ask for a simple JSON object.
  const instruction = `Write the emoji. Use very complicated words and be very verbose. It must precisely follow the sample. Only respond with valid JSON object containing a valid JSON array named 'list', in this format:
  {"list":["😭: sobbing face", "🍎: red apple"]}`;

  const model = models.getModel<OpenAIChatModel>(generationModelName);
  const input = model.createInput([
    new SystemMessage(instruction),
    new UserMessage(text),
  ]);

  input.responseFormat = ResponseFormat.Json;

  const output = model.invoke(input);

  // The output should contain the JSON string we asked for.
  const json = output.choices[0].message.content.trim();

  const results = JSON.parse<Map<string, string[]>>(json);
  return results.get("list");
}

export function upsertAllStarterEmojis(): string {
  const generateBatchSize: i32 = 10;
  const upsertBatchSize: i32 = 50;
  let emojiDescriptionList: string[] = [];

  // Generate emoji descriptions in batches of 10
  for (let i: i32 = 0; i < starterEmojis.length; i += generateBatchSize) {
    const end: i32 = min(i + generateBatchSize, starterEmojis.length);
    const batch: string[] = starterEmojis.slice(i, end);
    emojiDescriptionList = emojiDescriptionList.concat(
      generateListForEmoji(batch.join(", ")),
    );
  }

  const emojisList: string[] = [];
  for (let i: i32 = 0; i < emojiDescriptionList.length; i++) {
    const description = emojiDescriptionList[i];
    emojisList.push(getEmojiFromString(description));
    const colonIndex = description.indexOf(":");
    if (colonIndex !== -1) {
      emojiDescriptionList[i] = description.substring(colonIndex + 1).trim();
    }
  }

  // Upsert emojis in batches of 50
  for (let i: i32 = 0; i < emojiDescriptionList.length; i += upsertBatchSize) {
    const end: i32 = min(i + upsertBatchSize, emojiDescriptionList.length);
    const descriptionBatch: string[] = emojiDescriptionList.slice(i, end);
    const emojiBatch: string[] = emojisList.slice(i, end);
    const response = collections.upsertBatch(
      emojis,
      emojiBatch,
      descriptionBatch,
    );
    if (!response.isSuccessful) {
      return response.error;
    }
  }

  return "All starter emojis upserted successfully";
}

function generateText(instruction: string, prompt: string): string {
  // The imported ChatModel interface follows the OpenAI Chat completion model input format.
  const model = models.getModel<OpenAIChatModel>(generationModelName);
  const input = model.createInput([
    new SystemMessage(instruction),
    new UserMessage(prompt),
  ]);

  // Here we invoke the model with the input we created.
  const output = model.invoke(input);

  // The output is also specific to the ChatModel interface.
  // Here we return the trimmed content of the first choice.
  return output.choices[0].message.content.trim();
}

export function upsertEmoji(emoji: string): string {
  const emojiDescription = generateText(
    "generate a concise one sentence description for the emoji sent",
    emoji,
  );
  const response = collections.upsert(emojis, emoji, emojiDescription);
  if (!response.isSuccessful) {
    throw new Error(response.error);
  }
  return response.status;
}

export function getEmojiDescription(emoji: string): string {
  return collections.getText(emojis, emoji);
}

export function findMatchingEmoji(
  emojiDescription: string,
): collections.CollectionSearchResult {
  return collections.search(emojis, searchMethod, emojiDescription, 5, true);
}

export function recomputeIndex(): string {
  const response = collections.recomputeSearchMethod(emojis, searchMethod);
  if (!response.isSuccessful) {
    throw new Error(response.error);
  }
  return response.status;
}
