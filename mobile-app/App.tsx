import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  Platform,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';

type PredictionResponse = {
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
};

const MODEL_OPTIONS = [
  { label: 'TorchScript Scripted', value: 'scripted' },
  { label: 'TorchScript Traced', value: 'traced' },
  { label: 'PyTorch Weights', value: 'pth' },
] as const;

const DEFAULT_API_BASE_URL = Platform.select({
  android: 'http://10.0.2.2:8000',
  ios: 'http://127.0.0.1:8000',
  default: 'http://127.0.0.1:8000',
});

const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL;

export default function App() {
  const [selectedModel, setSelectedModel] = useState<(typeof MODEL_OPTIONS)[number]['value']>('scripted');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('Pick a tyre image to start the inspection.');

  const modelLabel = useMemo(
    () => MODEL_OPTIONS.find((option) => option.value === selectedModel)?.label ?? selectedModel,
    [selectedModel],
  );

  async function pickImage() {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
      base64: true,
    });

    if (result.canceled || !result.assets.length) {
      return;
    }

    const asset = result.assets[0];
    setImageUri(asset.uri);
    setImageBase64(asset.base64 ?? null);
    setPrediction(null);
    setMessage('Image selected. Run prediction to classify the tyre.');
  }

  async function runPrediction() {
    if (!imageBase64) {
      setMessage('Select an image first.');
      return;
    }

    setLoading(true);
    setMessage(`Sending image to the model service at ${API_BASE_URL}...`);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: imageBase64,
          model_name: selectedModel,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Prediction request failed.');
      }

      const data = (await response.json()) as PredictionResponse;
      setPrediction(data);
      setMessage(`Prediction complete using ${modelLabel}.`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setPrediction(null);
      setMessage(`Prediction failed: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <LinearGradient colors={['#07111f', '#0f1d31', '#16314f']} style={styles.screen}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.heroCard}>
          <Text style={styles.kicker}>TyreNet Mobile</Text>
          <Text style={styles.title}>Tyre defect classification on Android and web</Text>
          <Text style={styles.subtitle}>
            A single Expo interface for picking a tyre image, selecting a model export, and
            running the same prediction flow on mobile or in the browser.
          </Text>
        </View>

        <View style={styles.panel}>
          <Text style={styles.sectionLabel}>Model export</Text>
          <View style={styles.optionRow}>
            {MODEL_OPTIONS.map((option) => {
              const active = option.value === selectedModel;
              return (
                <Pressable
                  key={option.value}
                  onPress={() => setSelectedModel(option.value)}
                  style={({ pressed }) => [
                    styles.optionChip,
                    active && styles.optionChipActive,
                    pressed && styles.optionChipPressed,
                  ]}
                >
                  <Text style={[styles.optionLabel, active && styles.optionLabelActive]}>
                    {option.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>

          <View style={styles.actionRow}>
            <Pressable onPress={pickImage} style={({ pressed }) => [styles.primaryButton, pressed && styles.buttonPressed]}>
              <Text style={styles.primaryButtonText}>Choose Image</Text>
            </Pressable>
            <Pressable
              onPress={runPrediction}
              disabled={loading}
              style={({ pressed }) => [
                styles.secondaryButton,
                pressed && !loading && styles.buttonPressed,
                loading && styles.buttonDisabled,
              ]}
            >
              {loading ? <ActivityIndicator color="#0a1220" /> : <Text style={styles.secondaryButtonText}>Run Prediction</Text>}
            </Pressable>
          </View>
        </View>

        <View style={styles.previewGrid}>
          <View style={styles.previewCard}>
            <Text style={styles.sectionLabel}>Image preview</Text>
            {imageUri ? (
              <Image source={{ uri: imageUri }} style={styles.previewImage} resizeMode="cover" />
            ) : (
              <View style={styles.placeholderBox}>
                <Text style={styles.placeholderTitle}>No image selected</Text>
                <Text style={styles.placeholderText}>Pick a tyre photo from your gallery to begin.</Text>
              </View>
            )}
          </View>

          <View style={styles.previewCard}>
            <Text style={styles.sectionLabel}>Result</Text>
            <View style={styles.resultBox}>
              <Text style={styles.resultTitle}>{prediction ? prediction.class_name : 'Awaiting prediction'}</Text>
              <Text style={styles.resultText}>{message}</Text>
              {prediction ? (
                <>
                  <Text style={styles.confidenceText}>{prediction.confidence.toFixed(2)}%</Text>
                  <View style={styles.metricList}>
                    {Object.entries(prediction.probabilities).map(([key, value]) => (
                      <View key={key} style={styles.metricRow}>
                        <Text style={styles.metricLabel}>{key}</Text>
                        <Text style={styles.metricValue}>{(value * 100).toFixed(2)}%</Text>
                      </View>
                    ))}
                  </View>
                </>
              ) : null}
            </View>
          </View>
        </View>

        <View style={styles.footerCard}>
          <Text style={styles.footerTitle}>Connection</Text>
          <Text style={styles.footerText}>API base URL: {API_BASE_URL}</Text>
          <Text style={styles.footerText}>Selected model: {modelLabel}</Text>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    gap: 16,
    paddingTop: 64,
    paddingBottom: 36,
  },
  heroCard: {
    borderRadius: 28,
    padding: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  kicker: {
    color: '#8fd3ff',
    textTransform: 'uppercase',
    letterSpacing: 2,
    fontSize: 12,
    marginBottom: 8,
    fontWeight: '700',
  },
  title: {
    color: '#f8fbff',
    fontSize: 32,
    lineHeight: 38,
    fontWeight: '800',
    marginBottom: 12,
  },
  subtitle: {
    color: 'rgba(248, 251, 255, 0.82)',
    fontSize: 15,
    lineHeight: 22,
  },
  panel: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: 'rgba(7, 14, 27, 0.72)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  sectionLabel: {
    color: '#a8c7ff',
    fontSize: 13,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
    marginBottom: 12,
  },
  optionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  optionChip: {
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 999,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  optionChipActive: {
    backgroundColor: '#f6c85f',
    borderColor: '#f6c85f',
  },
  optionChipPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.98 }],
  },
  optionLabel: {
    color: '#d7e7ff',
    fontWeight: '700',
  },
  optionLabelActive: {
    color: '#08111f',
  },
  actionRow: {
    marginTop: 18,
    flexDirection: 'row',
    gap: 12,
  },
  primaryButton: {
    flex: 1,
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: 'center',
    backgroundColor: '#8fd3ff',
  },
  secondaryButton: {
    flex: 1,
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: 'center',
    backgroundColor: '#f6c85f',
  },
  secondaryButtonText: {
    color: '#0a1220',
    fontWeight: '800',
  },
  primaryButtonText: {
    color: '#0a1220',
    fontWeight: '800',
  },
  buttonPressed: {
    opacity: 0.9,
    transform: [{ scale: 0.985 }],
  },
  buttonDisabled: {
    opacity: 0.75,
  },
  previewGrid: {
    gap: 16,
  },
  previewCard: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  previewImage: {
    width: '100%',
    height: 260,
    borderRadius: 18,
    backgroundColor: '#0b1528',
  },
  placeholderBox: {
    minHeight: 260,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'rgba(8, 17, 31, 0.65)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  placeholderTitle: {
    color: '#f8fbff',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
  },
  placeholderText: {
    color: 'rgba(248, 251, 255, 0.72)',
    textAlign: 'center',
    lineHeight: 21,
  },
  resultBox: {
    minHeight: 260,
    borderRadius: 18,
    padding: 16,
    backgroundColor: 'rgba(8, 17, 31, 0.65)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  resultTitle: {
    color: '#f8fbff',
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 8,
  },
  resultText: {
    color: 'rgba(248, 251, 255, 0.78)',
    lineHeight: 21,
    marginBottom: 14,
  },
  confidenceText: {
    color: '#8fd3ff',
    fontSize: 32,
    fontWeight: '900',
    marginBottom: 16,
  },
  metricList: {
    gap: 10,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  metricLabel: {
    color: '#dfeaff',
    fontWeight: '700',
  },
  metricValue: {
    color: '#f6c85f',
    fontWeight: '800',
  },
  footerCard: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  footerTitle: {
    color: '#f8fbff',
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 8,
  },
  footerText: {
    color: 'rgba(248, 251, 255, 0.78)',
    lineHeight: 20,
  },
});
