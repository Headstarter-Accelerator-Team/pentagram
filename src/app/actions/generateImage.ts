"use server";

export async function generateImage(text: string) {
  try {
    const response = await fetch(
      "http://10.91.174.20:3000/api/generate-image",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API_KEY": process.env.API_KEY || "",
        },
        body: JSON.stringify({ text }),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to generate image HTTP error, status: ${response.status}`
      );
    }
    const data = await response.json();
    return data;
  } catch (err) {
    console.error("Server error: ", err);
    return {
      success: false,
      error: err instanceof Error ? err.message : "Failed to generate Image",
    };
  }
}
