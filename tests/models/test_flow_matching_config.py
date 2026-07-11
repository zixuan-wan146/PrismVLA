import unittest


class FlowMatchingConfigTests(unittest.TestCase):
    def test_action_head_can_be_constructed_without_config(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
        )

        self.assertIsInstance(head, torch.nn.Module)
        self.assertEqual(head.horizon, 2)
        self.assertEqual(head.per_action_dim, 3)
        self.assertEqual(head.action_dim, 6)

    def test_action_encoder_rejects_wrong_horizon(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        encoder = flow_matching.MultiEmbodimentActionEncoder(
            action_dim=3,
            embed_dim=8,
            hidden_dim=8,
            horizon=2,
            num_categories=1,
        )

        action_seq = torch.zeros(1, 3, 3)
        category_id = torch.zeros(1, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "must match horizon"):
            encoder(action_seq, category_id)

    def test_action_head_rejects_wrong_training_action_mask_shape(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
        )
        fused_tokens = torch.zeros(1, 1, 8)
        actions_gt = torch.zeros(1, 2, 3)
        action_mask = torch.ones(1, 6)

        with self.assertRaisesRegex(ValueError, "action_mask shape"):
            head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

    def test_single_step_action_head_registers_all_parameters_before_forward(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=3,
            horizon=1,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
        )

        param_names_before = set(dict(head.named_parameters()))
        self.assertIn("action_encoder.W1.linear.weight", param_names_before)
        self.assertIn("action_decoder.W3.linear.weight", param_names_before)

        fused_tokens = torch.zeros(2, 1, 8)
        actions_gt = torch.zeros(2, 1, 3)
        action_mask = torch.ones(2, 1, 3)
        pred_velocity, noise = head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

        self.assertEqual(tuple(pred_velocity.shape), (2, 3))
        self.assertEqual(tuple(noise.shape), (2, 1, 3))
        self.assertEqual(set(dict(head.named_parameters())), param_names_before)

    def test_inference_keeps_masked_action_dimensions_zero_after_final_step(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        torch.manual_seed(0)
        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=2,
        )
        fused_tokens = torch.zeros(1, 1, 8)
        action_mask = torch.tensor([[1.0, 0.0, 1.0]])

        action = head.get_action(fused_tokens, action_mask=action_mask).view(1, 2, 3)

        self.assertTrue(torch.equal(action[:, :, 1], torch.zeros_like(action[:, :, 1])))

    def test_inference_uses_midpoint_tau_grid(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=3,
            horizon=1,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=2,
        )
        observed = []
        original_time_embedding = head._time_embedding

        def capture_time_embedding(t, *, device, dtype):
            observed.append(t.detach().cpu())
            return original_time_embedding(t, device=device, dtype=dtype)

        head._time_embedding = capture_time_embedding
        fused_tokens = torch.zeros(1, 1, 8)
        action_mask = torch.ones(1, 3)

        head.get_action(fused_tokens, action_mask=action_mask)

        self.assertEqual(len(observed), 2)
        self.assertTrue(torch.allclose(observed[0], torch.tensor([0.25])))
        self.assertTrue(torch.allclose(observed[1], torch.tensor([0.75])))

    def test_action_head_training_shapes_for_common_horizons(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        for horizon, per_action_dim in ((1, 7), (14, 7), (16, 8)):
            with self.subTest(horizon=horizon, per_action_dim=per_action_dim):
                action_dim = horizon * per_action_dim
                head = flow_matching.FlowmatchingActionHead(
                    embed_dim=8,
                    hidden_dim=16,
                    action_dim=action_dim,
                    horizon=horizon,
                    per_action_dim=per_action_dim,
                    num_heads=2,
                    num_layers=1,
                    num_inference_timesteps=1,
                )
                fused_tokens = torch.zeros(2, 3, 8)
                actions_gt = torch.zeros(2, horizon, per_action_dim)
                action_mask = torch.ones(2, horizon, per_action_dim)

                pred_velocity, noise = head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

                self.assertEqual(tuple(pred_velocity.shape), (2, action_dim))
                self.assertEqual(tuple(noise.shape), (2, horizon, per_action_dim))

    def test_multi_category_action_head_training_shape(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
            num_categories=2,
        )
        fused_tokens = torch.zeros(2, 3, 8)
        actions_gt = torch.zeros(2, 2, 3)
        action_mask = torch.ones(2, 2, 3)
        embodiment_id = torch.tensor([0, 1], dtype=torch.long)

        pred_velocity, noise = head(
            fused_tokens,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
        )

        self.assertEqual(tuple(pred_velocity.shape), (2, 6))
        self.assertEqual(tuple(noise.shape), (2, 2, 3))

    def test_direct_bridge_action_head_accepts_all_context_sources(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        torch.manual_seed(0)
        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=2,
            num_inference_timesteps=1,
        )
        fused_tokens = torch.zeros(2, 5, 8)
        hidden_states = [torch.randn(2, 5, 8) for _ in range(4)]
        short_memory = torch.randn(2, 6, 8)
        short_time_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        short_mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])
        plan_tokens = torch.randn(2, 1, 8)
        state = torch.randn(2, 7)
        actions_gt = torch.zeros(2, 2, 3)
        action_mask = torch.ones(2, 2, 3)

        pred_velocity, noise = head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            vlm_hidden_states=hidden_states,
            short_memory_tokens=short_memory,
            short_memory_time_ids=short_time_ids,
            short_memory_mask=short_mask,
            plan_tokens=plan_tokens,
        )

        self.assertEqual(tuple(pred_velocity.shape), (2, 6))
        self.assertEqual(tuple(noise.shape), (2, 2, 3))
        self.assertEqual(tuple(head.plan_slot_embeddings.shape), (8, 8))

    def test_direct_bridge_action_head_inference_smoke_with_all_context_sources(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        torch.manual_seed(0)
        head = flow_matching.FlowmatchingActionHead(
            embed_dim=16,
            hidden_dim=32,
            action_dim=8,
            horizon=4,
            per_action_dim=2,
            num_heads=4,
            num_layers=2,
            num_inference_timesteps=2,
        )
        fused_tokens = torch.randn(1, 7, 16)
        hidden_states = [torch.randn(1, 7, 16) for _ in range(4)]
        short_memory = torch.randn(1, 8, 16)
        short_time_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        short_mask = torch.tensor([[True, True, True, True, True, True, False, False]])
        plan_tokens = torch.randn(1, 1, 16)
        state = torch.randn(1, 7)
        action_mask = torch.tensor([[1.0, 0.0]])

        action = head.get_action(
            fused_tokens,
            state=state,
            action_mask=action_mask,
            vlm_hidden_states=hidden_states,
            short_memory_tokens=short_memory,
            short_memory_time_ids=short_time_ids,
            short_memory_mask=short_mask,
            plan_tokens=plan_tokens,
        ).view(1, 4, 2)

        self.assertEqual(tuple(action.shape), (1, 4, 2))
        self.assertTrue(torch.equal(action[:, :, 1], torch.zeros_like(action[:, :, 1])))

    def test_short_memory_time_ids_reject_out_of_range_values(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("prism.models.action_head")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
            short_memory_time_bins=2,
        )

        with self.assertRaisesRegex(ValueError, "short_memory_time_ids"):
            head._prepare_short_memory_time_ids(
                torch.tensor([[0, 1, 2]]),
                batch_size=1,
                token_count=3,
                device=torch.device("cpu"),
            )

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
