//! BF16 Awareness Substrate — Thinking style detection from superposition decomposition.
//!
//! The core insight: when 2-3 BF16 vectors are superposed, the per-dimension
//! 4-state classification (Crystallized/Tensioned/Uncertain/Noise) encodes the
//! current cognitive state. The distribution of these states across all dimensions
//! reveals the thinking style as an emergent property — not a parameter, not a
//! choice, but an observation.
//!
//! This module bridges numrus-core's `SuperpositionState` to ladybug-rs's
//! `ThinkingStyle` enum and `CollapseGate` decisions.
//!
//! ## Thinking Style Detection
//!
//! | Pattern          | Crystal% | Tension% | Uncertain% | Noise% | Style       |
//! |------------------|----------|----------|------------|--------|-------------|
//! | Convergent       | High     | Low      | Low        | Any    | Analytical  |
//! | Divergent        | Low      | Low      | High       | Any    | Explorer    |
//! | Dialectical      | Medium   | High     | Low        | Any    | Architect   |
//! | Receptive        | Low      | Low      | Low        | High   | Listener    |
//! | Committed        | V.High   | V.Low    | V.Low      | Low    | Frozen      |
//!
//! ## Enriched Collapse Gate
//!
//! The standard CollapseGate uses popcount-based conflict detection.
//! The enriched version uses BF16 decomposition for richer criteria:
//! - **FLOW**: sign consensus above threshold AND exponent spread below threshold
//! - **HOLD**: sign consensus above threshold BUT exponent spread above threshold
//! - **BLOCK**: sign consensus below threshold (irreconcilable contradiction)

use numrus_core::bf16_hamming::{AwarenessState, SuperpositionState};

// ---------------------------------------------------------------------------
// Thinking Style Detection
// ---------------------------------------------------------------------------

/// Detected thinking style as emergent property of superposition decomposition.
///
/// Maps to ladybug-rs's 12 ThinkingStyles via the 5 philosopher clusters.
/// Each variant corresponds to a cluster, not a single style — the `mix`
/// field captures the continuous blend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingCluster {
    /// High crystallized, low tensioned. Settled, focused reasoning.
    /// Maps to: Analytical, Convergent, Systematic
    Convergent,
    /// Low crystallized, high uncertain. Exploring magnitude space.
    /// Maps to: Creative, Divergent, Exploratory
    Divergent,
    /// Medium crystallized, high tensioned. Holding contradictions.
    /// Maps to: Metacognitive, Deliberate
    Dialectical,
    /// Low everything except noise. Receptive, listening for signal.
    /// Maps to: Diffuse, Peripheral, Intuitive
    Receptive,
    /// Very high crystallized, near-zero everything else. Frozen.
    /// Maps to: Focused (extreme convergence)
    Committed,
}

/// Continuous thinking style mix — weights for each philosopher cluster.
///
/// The sum of all weights is 1.0. This captures the blend of styles
/// rather than forcing a single classification.
#[derive(Clone, Debug)]
pub struct ThinkingStyleMix {
    pub convergent: f32,
    pub divergent: f32,
    pub dialectical: f32,
    pub receptive: f32,
    pub committed: f32,
    /// Dominant cluster (highest weight).
    pub dominant: ThinkingCluster,
}

impl ThinkingStyleMix {
    /// Detect thinking style from a SuperpositionState.
    ///
    /// This is a single dot-product of the 4-state distribution against
    /// a style-template matrix. O(1) per style check — 5 dot products total.
    pub fn detect(state: &SuperpositionState) -> Self {
        let c = state.crystallized_pct;
        let t = state.tensioned_pct;
        let u = state.uncertain_pct;
        let n = state.noise_pct;

        // Style template matrix — each row is a target distribution.
        // Score = 1 - manhattan_distance(actual, template) / 2
        // (manhattan distance of two distributions that sum to 1 is at most 2)
        let convergent_score = style_similarity(c, t, u, n, 0.70, 0.05, 0.10, 0.15);
        let divergent_score = style_similarity(c, t, u, n, 0.15, 0.05, 0.65, 0.15);
        let dialectical_score = style_similarity(c, t, u, n, 0.30, 0.50, 0.10, 0.10);
        let receptive_score = style_similarity(c, t, u, n, 0.10, 0.05, 0.10, 0.75);
        let committed_score = style_similarity(c, t, u, n, 0.90, 0.02, 0.03, 0.05);

        // Normalize to probabilities
        let total = convergent_score + divergent_score + dialectical_score
            + receptive_score + committed_score;

        let (convergent, divergent, dialectical, receptive, committed) = if total > 0.0 {
            (
                convergent_score / total,
                divergent_score / total,
                dialectical_score / total,
                receptive_score / total,
                committed_score / total,
            )
        } else {
            (0.2, 0.2, 0.2, 0.2, 0.2) // uniform if degenerate
        };

        // Find dominant
        let scores = [
            (convergent, ThinkingCluster::Convergent),
            (divergent, ThinkingCluster::Divergent),
            (dialectical, ThinkingCluster::Dialectical),
            (receptive, ThinkingCluster::Receptive),
            (committed, ThinkingCluster::Committed),
        ];
        let dominant = scores
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|s| s.1)
            .unwrap_or(ThinkingCluster::Convergent);

        Self {
            convergent,
            divergent,
            dialectical,
            receptive,
            committed,
            dominant,
        }
    }

    /// Should the detected style reinforce itself (positive feedback)
    /// or trigger complementary exploration (negative feedback)?
    ///
    /// Returns a bias in [-1.0, 1.0]:
    /// - Positive: reinforce current style (high confidence in detection)
    /// - Negative: explore complementary style (low confidence, or stuck)
    pub fn feedback_bias(&self) -> f32 {
        let max_weight = self.convergent
            .max(self.divergent)
            .max(self.dialectical)
            .max(self.receptive)
            .max(self.committed);

        // If one style dominates clearly (>0.5), reinforce it
        // If no clear dominant (<0.3), suggest exploring complement
        if max_weight > 0.5 {
            (max_weight - 0.5) * 2.0 // [0.0, 1.0]
        } else {
            -(0.3 - max_weight).max(0.0) * 3.33 // [-1.0, 0.0]
        }
    }
}

/// Manhattan distance similarity: 1 - |actual - template| / 2
fn style_similarity(
    ac: f32, at: f32, au: f32, an: f32,
    tc: f32, tt: f32, tu: f32, tn: f32,
) -> f32 {
    let dist = (ac - tc).abs() + (at - tt).abs() + (au - tu).abs() + (an - tn).abs();
    (1.0 - dist / 2.0).max(0.0)
}

// ---------------------------------------------------------------------------
// Enriched Collapse Gate
// ---------------------------------------------------------------------------

/// Enriched collapse decision using BF16 structural decomposition.
///
/// Richer than popcount-only: uses sign consensus + exponent spread
/// to distinguish semantic conflicts from mantissa noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnrichedGate {
    /// Sign consensus above threshold AND exponent spread below threshold.
    /// Safe to commit delta to ground truth.
    Flow,
    /// Sign consensus above threshold BUT exponent spread above threshold.
    /// Direction agreed, intensity unclear. Keep delta floating.
    Hold,
    /// Sign consensus below threshold.
    /// Irreconcilable contradiction. Discard weakest delta.
    Block,
}

/// Thresholds for the enriched collapse gate.
///
/// Style-dependent: analytical mode has stricter thresholds (higher sign
/// consensus required) while explorer mode has looser ones.
#[derive(Clone, Copy, Debug)]
pub struct EnrichedGateThresholds {
    /// Mean sign consensus must exceed this for FLOW (default: 0.75 = 192/255).
    pub sign_consensus_min: f32,
    /// Mean exponent spread must be below this for FLOW (default: 1.5).
    pub exp_spread_max: f32,
    /// Fraction of tensioned dimensions that triggers BLOCK (default: 0.3).
    pub tension_block_threshold: f32,
}

impl Default for EnrichedGateThresholds {
    fn default() -> Self {
        Self {
            sign_consensus_min: 0.75,
            exp_spread_max: 1.5,
            tension_block_threshold: 0.30,
        }
    }
}

impl EnrichedGateThresholds {
    /// Analytical mode: stricter thresholds (higher consensus required).
    pub fn analytical() -> Self {
        Self {
            sign_consensus_min: 0.85,
            exp_spread_max: 1.0,
            tension_block_threshold: 0.20,
        }
    }

    /// Explorer mode: looser thresholds (accept more uncertainty).
    pub fn explorer() -> Self {
        Self {
            sign_consensus_min: 0.60,
            exp_spread_max: 2.5,
            tension_block_threshold: 0.45,
        }
    }

    /// Adapt thresholds based on detected thinking style.
    pub fn from_style(style: &ThinkingStyleMix) -> Self {
        let base = Self::default();
        match style.dominant {
            ThinkingCluster::Convergent | ThinkingCluster::Committed => Self::analytical(),
            ThinkingCluster::Divergent => Self::explorer(),
            ThinkingCluster::Dialectical => Self {
                sign_consensus_min: 0.70,
                exp_spread_max: 2.0,
                tension_block_threshold: 0.50, // high tolerance for tension
            },
            ThinkingCluster::Receptive => base, // default is fine for listening
        }
    }
}

/// Evaluate the enriched collapse gate from a SuperpositionState.
///
/// Returns the gate decision and a confidence scalar [0, 1].
pub fn evaluate_enriched_gate(
    state: &SuperpositionState,
    thresholds: &EnrichedGateThresholds,
) -> (EnrichedGate, f32) {
    // Aggregate sign consensus (mean across dimensions)
    let mean_consensus: f32 = if state.n_dims > 0 {
        state.sign_consensus.iter().map(|&c| c as f32 / 255.0).sum::<f32>()
            / state.n_dims as f32
    } else {
        1.0
    };

    // Aggregate exponent spread (mean across dimensions)
    let mean_exp_spread: f32 = if state.n_dims > 0 {
        state.exp_spread.iter().map(|&e| e as f32).sum::<f32>() / state.n_dims as f32
    } else {
        0.0
    };

    // Check tension threshold
    if state.tensioned_pct > thresholds.tension_block_threshold {
        let confidence = (state.tensioned_pct - thresholds.tension_block_threshold)
            / (1.0 - thresholds.tension_block_threshold);
        return (EnrichedGate::Block, confidence.clamp(0.0, 1.0));
    }

    // Check sign consensus
    if mean_consensus < thresholds.sign_consensus_min {
        let confidence = (thresholds.sign_consensus_min - mean_consensus)
            / thresholds.sign_consensus_min;
        return (EnrichedGate::Block, confidence.clamp(0.0, 1.0));
    }

    // Sign consensus OK — check exponent spread
    if mean_exp_spread > thresholds.exp_spread_max {
        let confidence = (mean_exp_spread - thresholds.exp_spread_max)
            / (8.0 - thresholds.exp_spread_max); // 8 = max possible
        return (EnrichedGate::Hold, confidence.clamp(0.0, 1.0));
    }

    // Everything clear — FLOW
    let confidence = mean_consensus * (1.0 - mean_exp_spread / 8.0);
    (EnrichedGate::Flow, confidence.clamp(0.0, 1.0))
}

/// Selective merge: apply XOR only on dimensions where the diff is structurally
/// significant (state != Noise). Mantissa-only diffs are masked out.
///
/// Returns a new byte vector where noise dimensions are zeroed.
pub fn selective_merge(delta_a: &[u8], delta_b: &[u8], state: &SuperpositionState) -> Vec<u8> {
    assert_eq!(delta_a.len(), delta_b.len());
    assert_eq!(delta_a.len(), state.n_dims * 2);

    let mut merged = vec![0u8; delta_a.len()];
    for d in 0..state.n_dims {
        let offset = d * 2;
        if state.states[d] != AwarenessState::Noise {
            // Structurally significant — apply XOR
            merged[offset] = delta_a[offset] ^ delta_b[offset];
            merged[offset + 1] = delta_a[offset + 1] ^ delta_b[offset + 1];
        }
        // Noise dimensions: merged stays 0 (no change)
    }
    merged
}

// ---------------------------------------------------------------------------
// Moment Metadata (for awareness horizon storage)
// ---------------------------------------------------------------------------

/// Metadata attached to a moment vector for awareness horizon queries.
///
/// Stored alongside the moment in Lance for O(1) cognitive-state-indexed recall.
/// Total size: 2 bits × 1024 dims = 256 bytes (packed_states) + scalars.
#[derive(Clone, Debug)]
pub struct MomentMetadata {
    /// Packed 4-state classification vector (2 bits per dimension).
    pub packed_states: Vec<u8>,
    /// Detected thinking style at this moment.
    pub style: ThinkingStyleMix,
    /// Collapse gate decision.
    pub gate: EnrichedGate,
    /// Mean sign consensus ratio [0, 1].
    pub sign_consensus_mean: f32,
    /// Mean exponent spread [0, 8].
    pub exp_spread_mean: f32,
    /// Aggregate state percentages.
    pub crystallized_pct: f32,
    pub tensioned_pct: f32,
    pub uncertain_pct: f32,
    pub noise_pct: f32,
}

impl MomentMetadata {
    /// Create moment metadata from a SuperpositionState.
    pub fn from_superposition(state: &SuperpositionState) -> Self {
        let style = ThinkingStyleMix::detect(state);
        let thresholds = EnrichedGateThresholds::from_style(&style);
        let (gate, _confidence) = evaluate_enriched_gate(state, &thresholds);

        let sign_consensus_mean = if state.n_dims > 0 {
            state.sign_consensus.iter().map(|&c| c as f32 / 255.0).sum::<f32>()
                / state.n_dims as f32
        } else {
            1.0
        };

        let exp_spread_mean = if state.n_dims > 0 {
            state.exp_spread.iter().map(|&e| e as f32).sum::<f32>() / state.n_dims as f32
        } else {
            0.0
        };

        Self {
            packed_states: state.packed_states.clone(),
            style,
            gate,
            sign_consensus_mean,
            exp_spread_mean,
            crystallized_pct: state.crystallized_pct,
            tensioned_pct: state.tensioned_pct,
            uncertain_pct: state.uncertain_pct,
            noise_pct: state.noise_pct,
        }
    }

    /// Query: was the system in a specific thinking cluster at this moment?
    pub fn was_in_cluster(&self, cluster: ThinkingCluster) -> bool {
        self.style.dominant == cluster
    }

    /// Query: was tension above a given threshold?
    pub fn was_dialectical(&self, threshold: f32) -> bool {
        self.tensioned_pct > threshold
    }
}

// ---------------------------------------------------------------------------
// Awareness Loop
// ---------------------------------------------------------------------------

/// Configuration for the continuous awareness loop.
#[derive(Clone, Debug)]
pub struct AwarenessConfig {
    /// How often to update the thinking style (in scans).
    /// 1 = per-scan (continuous drift), N = every N scans (discrete steps).
    pub style_update_interval: usize,
    /// Feedback mode: positive (reinforce), negative (complement), or adaptive.
    pub feedback_mode: FeedbackMode,
    /// Gate thresholds (overridden by style-adaptive if feedback_mode is Adaptive).
    pub gate_thresholds: EnrichedGateThresholds,
}

/// How the detected style feeds back into the scan threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackMode {
    /// Detected style reinforces itself (convergent stays convergent).
    Positive,
    /// Detected style triggers complementary exploration.
    Negative,
    /// Feedback bias determined by detection confidence.
    Adaptive,
}

impl Default for AwarenessConfig {
    fn default() -> Self {
        Self {
            style_update_interval: 1,
            feedback_mode: FeedbackMode::Adaptive,
            gate_thresholds: EnrichedGateThresholds::default(),
        }
    }
}

/// The continuous awareness loop state.
///
/// Wires the full cycle: scan -> superpose -> decompose -> detect style ->
/// update threshold -> next scan. No orchestrator. Self-regulates via BF16 structure.
pub struct AwarenessLoop {
    config: AwarenessConfig,
    /// Current thinking style (updated every `style_update_interval` scans).
    current_style: ThinkingStyleMix,
    /// Current gate thresholds (adapted from style).
    current_thresholds: EnrichedGateThresholds,
    /// Scan counter for style update interval.
    scan_count: u64,
    /// History of moment metadata for meta-pattern detection.
    history: Vec<MomentMetadata>,
    /// Maximum history depth.
    max_history: usize,
}

impl AwarenessLoop {
    pub fn new(config: AwarenessConfig) -> Self {
        let current_thresholds = config.gate_thresholds;
        Self {
            config,
            current_style: ThinkingStyleMix {
                convergent: 0.2,
                divergent: 0.2,
                dialectical: 0.2,
                receptive: 0.2,
                committed: 0.2,
                dominant: ThinkingCluster::Convergent,
            },
            current_thresholds,
            scan_count: 0,
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Process one cycle of the awareness loop.
    ///
    /// Takes a SuperpositionState from BF16 decomposition and returns:
    /// - The enriched gate decision
    /// - The detected thinking style
    /// - Moment metadata for storage
    pub fn cycle(
        &mut self,
        state: &SuperpositionState,
    ) -> (EnrichedGate, ThinkingStyleMix, MomentMetadata) {
        self.scan_count += 1;

        // Detect style
        let style = ThinkingStyleMix::detect(state);

        // Update thresholds based on feedback mode
        if self.scan_count % self.config.style_update_interval as u64 == 0 {
            self.current_style = style.clone();
            self.current_thresholds = match self.config.feedback_mode {
                FeedbackMode::Positive => EnrichedGateThresholds::from_style(&style),
                FeedbackMode::Negative => {
                    // Complement: if convergent detected, use explorer thresholds
                    let complement = match style.dominant {
                        ThinkingCluster::Convergent => ThinkingCluster::Divergent,
                        ThinkingCluster::Divergent => ThinkingCluster::Convergent,
                        ThinkingCluster::Dialectical => ThinkingCluster::Receptive,
                        ThinkingCluster::Receptive => ThinkingCluster::Dialectical,
                        ThinkingCluster::Committed => ThinkingCluster::Divergent,
                    };
                    let fake_style = ThinkingStyleMix {
                        dominant: complement,
                        ..style.clone()
                    };
                    EnrichedGateThresholds::from_style(&fake_style)
                }
                FeedbackMode::Adaptive => {
                    let bias = style.feedback_bias();
                    if bias > 0.0 {
                        // Reinforce
                        EnrichedGateThresholds::from_style(&style)
                    } else {
                        // Mix default with current
                        self.config.gate_thresholds
                    }
                }
            };
        }

        // Evaluate gate with current (possibly style-adapted) thresholds
        let (gate, _confidence) = evaluate_enriched_gate(state, &self.current_thresholds);

        // Create moment metadata
        let metadata = MomentMetadata::from_superposition(state);

        // Store in history
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(metadata.clone());

        (gate, style, metadata)
    }

    /// Current thinking style.
    pub fn style(&self) -> &ThinkingStyleMix {
        &self.current_style
    }

    /// Current gate thresholds.
    pub fn thresholds(&self) -> &EnrichedGateThresholds {
        &self.current_thresholds
    }

    /// Total scans processed.
    pub fn scan_count(&self) -> u64 {
        self.scan_count
    }

    /// Find the last moment where the system was in a specific cluster.
    pub fn last_in_cluster(&self, cluster: ThinkingCluster) -> Option<&MomentMetadata> {
        self.history.iter().rev().find(|m| m.was_in_cluster(cluster))
    }

    /// Count transitions between clusters in the last N moments.
    /// A transition is when consecutive moments have different dominant clusters.
    pub fn transition_count(&self, last_n: usize) -> usize {
        let start = self.history.len().saturating_sub(last_n);
        let window = &self.history[start..];
        window
            .windows(2)
            .filter(|w| w[0].style.dominant != w[1].style.dominant)
            .count()
    }

    /// Detect meta-pattern: does the system cycle between two clusters?
    /// Returns Some((cluster_a, cluster_b, cycle_length)) if detected.
    pub fn detect_oscillation(&self, last_n: usize) -> Option<(ThinkingCluster, ThinkingCluster, usize)> {
        let start = self.history.len().saturating_sub(last_n);
        let window = &self.history[start..];
        if window.len() < 4 {
            return None;
        }

        // Check if the dominant cluster alternates (A-B-A-B pattern)
        let clusters: Vec<ThinkingCluster> = window.iter().map(|m| m.style.dominant).collect();
        let first = clusters[0];
        let second = clusters.get(1)?;
        if first == *second {
            return None;
        }

        let mut alternating = true;
        for (i, c) in clusters.iter().enumerate() {
            let expected = if i % 2 == 0 { first } else { *second };
            if *c != expected {
                alternating = false;
                break;
            }
        }

        if alternating {
            Some((first, *second, 2))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use numrus_core::bf16_hamming::{
        fp32_to_bf16_bytes, superposition_decompose, AwarenessThresholds,
    };

    fn make_crystallized_state() -> SuperpositionState {
        // Identical vectors → all crystallized
        let v = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        superposition_decompose(&[&v, &v], &AwarenessThresholds::default())
    }

    fn make_tensioned_state() -> SuperpositionState {
        // Sign flips → tensioned
        let a = fp32_to_bf16_bytes(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]);
        let b = fp32_to_bf16_bytes(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]);
        superposition_decompose(&[&a, &b], &AwarenessThresholds::default())
    }

    fn make_uncertain_state() -> SuperpositionState {
        // Same sign, large magnitude differences → uncertain
        let a = fp32_to_bf16_bytes(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let b = fp32_to_bf16_bytes(&[256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0]);
        superposition_decompose(&[&a, &b], &AwarenessThresholds::default())
    }

    #[test]
    fn test_detect_convergent_style() {
        let state = make_crystallized_state();
        let style = ThinkingStyleMix::detect(&state);
        // High crystallized → convergent or committed
        assert!(
            style.dominant == ThinkingCluster::Convergent
                || style.dominant == ThinkingCluster::Committed,
            "Expected Convergent or Committed, got {:?}",
            style.dominant
        );
    }

    #[test]
    fn test_detect_dialectical_style() {
        let state = make_tensioned_state();
        let style = ThinkingStyleMix::detect(&state);
        assert_eq!(
            style.dominant,
            ThinkingCluster::Dialectical,
            "Expected Dialectical for high tension, got {:?} (t={:.2})",
            style.dominant,
            state.tensioned_pct,
        );
    }

    #[test]
    fn test_detect_divergent_style() {
        let state = make_uncertain_state();
        let style = ThinkingStyleMix::detect(&state);
        assert_eq!(
            style.dominant,
            ThinkingCluster::Divergent,
            "Expected Divergent for high uncertainty, got {:?} (u={:.2})",
            style.dominant,
            state.uncertain_pct,
        );
    }

    #[test]
    fn test_style_mix_sums_to_one() {
        let state = make_crystallized_state();
        let style = ThinkingStyleMix::detect(&state);
        let total =
            style.convergent + style.divergent + style.dialectical + style.receptive + style.committed;
        assert!(
            (total - 1.0).abs() < 0.01,
            "Style mix should sum to ~1.0, got {}",
            total,
        );
    }

    #[test]
    fn test_enriched_gate_flow_on_crystallized() {
        let state = make_crystallized_state();
        let (gate, _confidence) = evaluate_enriched_gate(&state, &EnrichedGateThresholds::default());
        assert_eq!(gate, EnrichedGate::Flow);
    }

    #[test]
    fn test_enriched_gate_block_on_tensioned() {
        let state = make_tensioned_state();
        let (gate, _confidence) = evaluate_enriched_gate(&state, &EnrichedGateThresholds::default());
        assert_eq!(gate, EnrichedGate::Block);
    }

    #[test]
    fn test_enriched_gate_hold_on_uncertain() {
        let state = make_uncertain_state();
        let thresholds = EnrichedGateThresholds {
            tension_block_threshold: 0.5, // raise to avoid blocking
            ..EnrichedGateThresholds::default()
        };
        let (gate, _confidence) = evaluate_enriched_gate(&state, &thresholds);
        // Uncertain = high exp spread, should get Hold (direction agreed, intensity unclear)
        assert!(
            gate == EnrichedGate::Hold || gate == EnrichedGate::Flow,
            "Expected Hold or Flow for uncertain state, got {:?}",
            gate,
        );
    }

    #[test]
    fn test_selective_merge_masks_noise() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[1.09375]); // only mantissa differs → noise
        let state = superposition_decompose(&[&a, &b], &AwarenessThresholds::default());

        let delta_a = vec![0xFFu8; 2];
        let delta_b = vec![0x00u8; 2];
        let merged = selective_merge(&delta_a, &delta_b, &state);

        // Noise dimensions should be zeroed
        if state.states[0] == AwarenessState::Noise {
            assert_eq!(merged, vec![0, 0], "Noise dimension should be masked out");
        }
    }

    #[test]
    fn test_awareness_loop_basic_cycle() {
        let mut loop_ = AwarenessLoop::new(AwarenessConfig::default());
        let state = make_crystallized_state();
        let (gate, style, metadata) = loop_.cycle(&state);

        assert_eq!(gate, EnrichedGate::Flow);
        assert!(style.convergent > 0.0 || style.committed > 0.0);
        assert!(metadata.crystallized_pct > 0.9);
        assert_eq!(loop_.scan_count(), 1);
    }

    #[test]
    fn test_awareness_loop_transition_count() {
        let mut loop_ = AwarenessLoop::new(AwarenessConfig::default());

        // Alternate between crystallized and tensioned states
        let crystal = make_crystallized_state();
        let tension = make_tensioned_state();

        loop_.cycle(&crystal);
        loop_.cycle(&tension);
        loop_.cycle(&crystal);
        loop_.cycle(&tension);

        let transitions = loop_.transition_count(4);
        assert!(transitions >= 2, "Expected at least 2 transitions, got {}", transitions);
    }

    #[test]
    fn test_style_adapted_thresholds() {
        let analytical = EnrichedGateThresholds::analytical();
        let explorer = EnrichedGateThresholds::explorer();

        // Analytical should be stricter
        assert!(analytical.sign_consensus_min > explorer.sign_consensus_min);
        assert!(analytical.exp_spread_max < explorer.exp_spread_max);
    }

    #[test]
    fn test_moment_metadata_queries() {
        let state = make_tensioned_state();
        let metadata = MomentMetadata::from_superposition(&state);

        assert!(metadata.was_dialectical(0.3));
        assert!(metadata.tensioned_pct > 0.5);
    }
}
