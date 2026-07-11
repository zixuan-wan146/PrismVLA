import unittest


class PerceptualVisualMemoryTests(unittest.TestCase):
    def test_compressor_accepts_non_square_visual_token_count(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        compressor = prism.PerceptualTokenCompressor(hidden_dim=8, memory_tokens=3, num_heads=2)
        visual_tokens = torch.randn(2, 10, 8)
        visual_mask = torch.ones(2, 10, dtype=torch.bool)

        compressed = compressor(visual_tokens, visual_token_mask=visual_mask)

        self.assertEqual(tuple(compressed.shape), (2, 3, 8))
        self.assertTrue(torch.isfinite(compressed).all())

    def test_bottleneck_se_compressor_pools_square_visual_grid(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        compressor = prism.BottleneckSETokenCompressor(
            hidden_dim=8,
            memory_tokens=4,
            bottleneck_ratio=2,
        )

        compressed = compressor(torch.randn(2, 16, 8))

        self.assertEqual(tuple(compressed.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(compressed).all())

    def test_bottleneck_se_compressor_rejects_non_square_visual_grid(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        compressor = prism.BottleneckSETokenCompressor(hidden_dim=8, memory_tokens=4)

        with self.assertRaisesRegex(ValueError, "square visual-token grids"):
            compressor(torch.randn(2, 10, 8))

    def test_fixed_recent_memory_uses_r_half_and_r_offsets(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.FixedRecentVisualMemory(
            hidden_dim=8,
            tokens_per_observation=4,
            offsets=(8, 16),
            num_heads=2,
        )
        output = memory(
            {
                8: torch.randn(2, 16, 8),
                16: torch.randn(2, 16, 8),
            }
        )

        self.assertEqual(output.offsets, (8, 16))
        self.assertEqual(tuple(output.tokens.shape), (2, 8, 8))
        self.assertEqual(output.mask.tolist(), [[True] * 8, [True] * 8])
        self.assertEqual(tuple(output.as_model_kwargs()["memory_context"].shape), (2, 8, 8))

    def test_fixed_recent_memory_masks_missing_offsets_and_rows(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.FixedRecentVisualMemory(
            hidden_dim=8,
            tokens_per_observation=4,
            offsets=(8, 16),
            num_heads=2,
        )
        visual_tokens = torch.randn(2, 16, 8)
        visual_mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )

        output = memory(
            {8: visual_tokens},
            visual_token_masks_by_offset={8: visual_mask},
            batch_size=2,
        )

        self.assertEqual(output.mask.tolist(), [[True, True, True, True, False, False, False, False], [False] * 8])
        self.assertFalse(torch.allclose(output.tokens[0, :4], torch.zeros_like(output.tokens[0, :4])))
        self.assertTrue(torch.allclose(output.tokens[0, 4:], torch.zeros_like(output.tokens[0, 4:])))
        self.assertTrue(torch.allclose(output.tokens[1], torch.zeros_like(output.tokens[1])))

    def test_fixed_recent_memory_accepts_sequence_inputs(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.FixedRecentVisualMemory(
            hidden_dim=8,
            tokens_per_observation=1,
            offsets=(8, 16),
            num_heads=2,
        )

        output = memory([torch.randn(1, 16, 8), None])

        self.assertEqual(tuple(output.tokens.shape), (1, 2, 8))
        self.assertEqual(output.mask.tolist(), [[True, False]])

    def test_bank_stores_first_frame_and_returns_bridge_kwargs(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.PerceptualVisualMemoryBank(
            hidden_dim=8,
            memory_tokens=2,
            capacity=4,
            num_heads=2,
            retrieval_layers=1,
        )

        output = memory(
            torch.randn(1, 5, 8),
            episode_ids=["ep0"],
            timesteps=[0],
        )

        self.assertEqual(tuple(output.tokens.shape), (1, 2, 8))
        self.assertEqual(output.mask.tolist(), [[True, True]])
        self.assertEqual(output.as_model_kwargs()["memory_context_mask"].tolist(), [[True, True]])
        self.assertEqual(len(memory.entries("ep0")), 1)
        self.assertTrue(torch.allclose(output.retrieved_tokens, torch.zeros_like(output.retrieved_tokens)))

    def test_second_frame_retrieves_history_without_growing_when_update_false(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.PerceptualVisualMemoryBank(
            hidden_dim=8,
            memory_tokens=2,
            capacity=4,
            num_heads=2,
            retrieval_layers=1,
        )
        memory(torch.randn(1, 5, 8), episode_ids=["ep0"], timesteps=[0])

        output = memory(
            torch.randn(1, 5, 8),
            episode_ids=["ep0"],
            timesteps=[1],
            update=False,
        )

        self.assertEqual(len(memory.entries("ep0")), 1)
        self.assertFalse(torch.allclose(output.retrieved_tokens, torch.zeros_like(output.retrieved_tokens)))
        self.assertTrue(((output.gate > 0.0) & (output.gate < 1.0)).all())

    def test_fifo_consolidation_keeps_latest_entries(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.PerceptualVisualMemoryBank(
            hidden_dim=8,
            memory_tokens=1,
            capacity=2,
            num_heads=2,
            retrieval_layers=0,
            consolidation="fifo",
        )
        for timestep in range(4):
            memory(torch.randn(1, 3, 8), episode_ids=["ep0"], timesteps=[timestep])

        self.assertEqual([entry.timestep for entry in memory.entries("ep0")], [2, 3])

    def test_merge_consolidation_keeps_capacity_and_clear_episode_works(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        memory = prism.PerceptualVisualMemoryBank(
            hidden_dim=8,
            memory_tokens=2,
            capacity=2,
            num_heads=2,
            retrieval_layers=0,
            consolidation="merge",
        )
        for timestep in range(5):
            memory(torch.randn(1, 4, 8), episode_ids=["ep0"], timesteps=[timestep])

        self.assertLessEqual(len(memory.entries("ep0")), 2)
        memory.clear_episode("ep0")
        self.assertEqual(memory.entries("ep0"), ())

    def test_rejects_empty_valid_token_row(self):
        torch = self._import_or_skip("torch")
        prism = self._import_or_skip("prism.models.memory")

        compressor = prism.PerceptualTokenCompressor(hidden_dim=8, memory_tokens=2, num_heads=2)

        with self.assertRaisesRegex(ValueError, "at least one valid visual token"):
            compressor(torch.randn(1, 4, 8), visual_token_mask=torch.zeros(1, 4, dtype=torch.bool))

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
