import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    // Request permission to access the camera roll
    getPermissionAsync();
  }, []);

  const getPermissionAsync = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      console.error('Permission to access camera roll denied');
    }
  };

  const selectImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });


      if (!result.cancelled) {
        // Send the image to Flask API
        await sendImageToAPI(result);
      }
    } catch (error) {
      console.error('Error selecting image:', error);
    }
  };

  const sendImageToAPI = async (image) => {
    const formData = new FormData();
    formData.append('image', {
      uri: image.uri,
      type: 'image/jpeg', // or 'image/png' based on your needs
      name: 'image.jpg', // or 'image.png'
    });

    

    try {
      //const response = await axios.post('http://127.0.0.1:5000/process_image', formData, {
        const response = await axios.post('http://34.121.212.193:5000/process_image', formData, {
  
      
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Display the processed image received from Flask API
      setSelectedImage(response.data.processed_image);
    } catch (error) {
      console.error('Error sending image to API:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Image Processing App</Text>
      <Button onPress={selectImage} title="Select Image" />
      {selectedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.subtitle}>Processed Image:</Text>
          <Image source={{ uri: selectedImage }} style={styles.image} />
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginVertical: 10,
  },
  imageContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 200,
    resizeMode: 'contain',
  },
});

export default App;
