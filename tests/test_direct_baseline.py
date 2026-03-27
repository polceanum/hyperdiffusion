"""Tests for direct baseline models."""
import pytest
import torch
import torch.nn as nn

from hyperdiffusion.direct_baseline import DirectPredictor, DirectTextProjector, DirectSystem


class TestDirectPredictor:
    """Tests for DirectPredictor model."""

    @pytest.fixture
    def model(self):
        return DirectPredictor(
            x_dim=2,
            y_dim=1,
            encoder_hidden=64,
            cond_dim=32,
            attention_heads=2,
            attention_layers=2,
        )

    def test_forward_basic(self, model):
        """Test basic forward pass."""
        batch_size, support_size, query_size = 2, 5, 3
        support_x = torch.randn(batch_size, support_size, 2)
        support_y = torch.randn(batch_size, support_size, 1)
        query_x = torch.randn(batch_size, query_size, 2)

        pred = model(support_x, support_y, query_x)
        
        assert pred.shape == (batch_size, query_size, 1)
        assert not torch.isnan(pred).any(), "Predictions contain NaN"

    def test_forward_different_batch_sizes(self, model):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            support_x = torch.randn(batch_size, 5, 2)
            support_y = torch.randn(batch_size, 5, 1)
            query_x = torch.randn(batch_size, 3, 2)
            
            pred = model(support_x, support_y, query_x)
            assert pred.shape == (batch_size, 3, 1)

    def test_forward_different_query_sizes(self, model):
        """Test forward pass with different query sizes."""
        batch_size, support_size = 2, 5
        support_x = torch.randn(batch_size, support_size, 2)
        support_y = torch.randn(batch_size, support_size, 1)
        
        for query_size in [1, 5, 10, 20]:
            query_x = torch.randn(batch_size, query_size, 2)
            pred = model(support_x, support_y, query_x)
            assert pred.shape == (batch_size, query_size, 1)

    def test_gradient_flow(self, model):
        """Test that gradients flow properly."""
        support_x = torch.randn(2, 5, 2, requires_grad=True)
        support_y = torch.randn(2, 5, 1, requires_grad=True)
        query_x = torch.randn(2, 3, 2, requires_grad=True)

        pred = model(support_x, support_y, query_x)
        loss = pred.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_device_handling(self, model):
        """Test that model works on CPU (and GPU if available)."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            model = model.to(device)
            support_x = torch.randn(2, 5, 2, device=device)
            support_y = torch.randn(2, 5, 1, device=device)
            query_x = torch.randn(2, 3, 2, device=device)

            pred = model(support_x, support_y, query_x)
            assert pred.device.type == device.split(":")[0]

    def test_deterministic_outputs(self, model):
        """Test that model produces consistent outputs with same input."""
        support_x = torch.randn(2, 5, 2)
        support_y = torch.randn(2, 5, 1)
        query_x = torch.randn(2, 3, 2)

        model.eval()
        with torch.no_grad():
            pred1 = model(support_x, support_y, query_x)
            pred2 = model(support_x, support_y, query_x)

        torch.testing.assert_close(pred1, pred2)

    def test_encode_support(self, model):
        """Test support set encoding."""
        support_x = torch.randn(2, 5, 2)
        support_y = torch.randn(2, 5, 1)

        context, latent = model.encoder(support_x, support_y)
        
        assert context.shape == (2, 32)  # cond_dim = 32
        assert latent.shape == (2, 32)   # latent_dim = cond_dim


class TestDirectTextProjector:
    """Tests for DirectTextProjector."""

    @pytest.fixture
    def projector(self):
        return DirectTextProjector(text_dim=768, cond_dim=64)

    def test_forward_basic(self, projector):
        """Test basic projection."""
        text_embed = torch.randn(2, 768)
        context = projector(text_embed)
        
        assert context.shape == (2, 64)
        assert not torch.isnan(context).any()

    def test_different_batch_sizes(self, projector):
        """Test with different batch sizes."""
        for batch_size in [1, 4, 8]:
            text_embed = torch.randn(batch_size, 768)
            context = projector(text_embed)
            assert context.shape == (batch_size, 64)

    def test_gradient_flow(self, projector):
        """Test gradient flow through projector."""
        text_embed = torch.randn(2, 768, requires_grad=True)
        context = projector(text_embed)
        loss = context.sum()
        loss.backward()
        
        assert text_embed.grad is not None


class TestDirectSystem:
    """Tests for DirectSystem."""

    @pytest.fixture
    def system(self):
        return DirectSystem(
            x_dim=2,
            y_dim=1,
            encoder_hidden=64,
            cond_dim=32,
            attention_heads=2,
            attention_layers=2,
            text_dim=256,
            text_mix_alpha=0.5,
        )

    def test_forward_with_support(self, system):
        """Test forward pass with support set."""
        batch_size, support_size, query_size = 2, 5, 3
        support_x = torch.randn(batch_size, support_size, 2)
        support_y = torch.randn(batch_size, support_size, 1)
        query_x = torch.randn(batch_size, query_size, 2)

        pred = system(support_x, support_y, query_x)
        
        assert pred.shape == (batch_size, query_size, 1)

    def test_forward_with_external_context(self, system):
        """Test forward pass with external context."""
        batch_size, query_size = 2, 3
        query_x = torch.randn(batch_size, query_size, 2)
        external_context = torch.randn(batch_size, 32)  # cond_dim = 32

        pred = system(
            support_x=None,
            support_y=None,
            query_x=query_x,
            context=external_context,
        )
        
        assert pred.shape == (batch_size, query_size, 1)

    def test_encode_support(self, system):
        """Test support set encoding."""
        support_x = torch.randn(2, 5, 2)
        support_y = torch.randn(2, 5, 1)
        
        context = system.encode_support(support_x, support_y)
        assert context.shape == (2, 32)

    def test_encode_text(self, system):
        """Test text encoding."""
        text_embed = torch.randn(2, 256)
        context = system.encode_text(text_embed)
        assert context.shape == (2, 32)

    def test_encode_oracle(self, system):
        """Test oracle encoding."""
        # Create one-hot vectors for 5 families
        one_hot = torch.zeros(2, 5)
        one_hot[0, 1] = 1
        one_hot[1, 3] = 1
        
        context = system.encode_oracle(one_hot)
        assert context.shape == (2, 32)

    def test_end_to_end_with_text(self, system):
        """Test end-to-end prediction with text modality."""
        batch_size, query_size = 2, 3
        text_embed = torch.randn(batch_size, 256)
        query_x = torch.randn(batch_size, query_size, 2)
        
        # Encode text and use as context
        context = system.encode_text(text_embed)
        pred = system(support_x=None, support_y=None, query_x=query_x, context=context)
        
        assert pred.shape == (batch_size, query_size, 1)

    def test_text_mix_alpha(self, system):
        """Test that text_mix_alpha is stored."""
        assert system.text_mix_alpha == 0.5


class TestDirectPredictorIntegration:
    """Integration tests for DirectPredictor."""

    def test_train_eval_modes(self):
        """Test train/eval mode switching."""
        model = DirectPredictor(x_dim=2, y_dim=1)
        
        # Train mode
        model.train()
        support_x = torch.randn(2, 5, 2)
        support_y = torch.randn(2, 5, 1)
        query_x = torch.randn(2, 3, 2)
        
        pred_train = model(support_x, support_y, query_x)
        
        # Eval mode (should be deterministic)
        model.eval()
        with torch.no_grad():
            pred_eval = model(support_x, support_y, query_x)
        
        assert pred_train.shape == pred_eval.shape

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = DirectPredictor(x_dim=2, y_dim=1, encoder_hidden=64, cond_dim=32)
        
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        # Rough sanity check: should be at least a few thousand params
        assert total_params > 1000

    def test_forward_backward_cycle(self):
        """Test complete training cycle."""
        model = DirectPredictor(x_dim=2, y_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Multiple steps
        for _ in range(3):
            support_x = torch.randn(2, 5, 2)
            support_y = torch.randn(2, 5, 1)
            query_x = torch.randn(2, 3, 2)
            query_y = torch.randn(2, 3, 1)
            
            pred = model(support_x, support_y, query_x)
            loss = torch.nn.functional.mse_loss(pred, query_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
