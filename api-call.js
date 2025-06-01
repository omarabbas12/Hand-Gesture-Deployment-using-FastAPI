function flattenLandmarks(landmarks) {
  const flattened = {};
  for (let i = 0; i < landmarks.length; i++) {
    flattened[`x${i + 1}`] = landmarks[i].x;
    flattened[`y${i + 1}`] = landmarks[i].y;
    flattened[`z${i + 1}`] = landmarks[i].z;
  }
  return flattened;
}

async function getPredictedLabel(processed_t) {
  try {
    // ✅ Flatten the landmarks before sending
    const formattedInput = flattenLandmarks(processed_t);

    const response = await fetch("http://34.227.25.110:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formattedInput),
    });

    if (!response.ok) {
      console.error("API error:", await response.text());
      return null;
    }

    const data = await response.json();
    console.log("Predicted label:", data.prediction);
    return data.prediction;
  } catch (error) {
    console.error("Fetch failed:", error);
    return null;
  }
}
