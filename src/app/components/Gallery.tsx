import React, { useEffect, useState } from "react";
import {
  Container,
  Card,
  CardMedia,
  Grid,
  CircularProgress,
} from "@mui/material";

const Gallery: React.FC<{ trigger: number }> = ({ trigger }) => {
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        // Fetch blobs from the "pentagram" folder

        // Extract image URLs (filter only image files)
        const response = await fetch("/api/get-images", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });
        if (!response.ok) {
          throw new Error(
            `Failed to generate image HTTP error, status: ${response.status}`
          );
        }
        const data = await response.json();
        console.log("data!!!!!!!11", data);
        setImages(data.imageURLs);
      } catch (error) {
        console.error("Error fetching images:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchImages();
  }, [trigger]);

  if (loading) {
    return (
      <Container
        maxWidth="lg"
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "50vh",
        }}
      >
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mb: 10 }}>
      <Grid container spacing={2}>
        {images.length === 0 ? (
          <p>No images found in the pentagram folder.</p>
        ) : (
          images.map((image, idx) => (
            <Grid item xs={12} sm={6} md={4} key={idx}>
              <Card sx={{ maxWidth: "100%" }}>
                <CardMedia
                  component="img"
                  image={image}
                  alt={`Image ${idx + 1}`}
                  sx={{
                    height: 300,
                    width: "100%",
                    objectFit: "cover",
                  }}
                />
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Container>
  );
};

export default Gallery;
