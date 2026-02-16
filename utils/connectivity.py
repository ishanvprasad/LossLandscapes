import torch
import torch.nn as nn
import copy

class BezierCurveModel(nn.Module):
    def __init__(self, model_start, model_end):
        super().__init__()
        # We freeze the start and end endpoints
        self.model_start = model_start
        self.model_end = model_end
        
        # We create a trainable "bend" (control point) initialized halfway between them
        self.bend_params = nn.ParameterDict()
        for (name, p_start), (_, p_end) in zip(model_start.named_parameters(), model_end.named_parameters()):
            if p_start.requires_grad:
                # Initialize the bend exactly in the middle
                middle_weight = (p_start.data + p_end.data) / 2.0
                self.bend_params[name.replace('.', '_')] = nn.Parameter(middle_weight)

    def get_interpolated_weights(self, t):
        """
        Quadratic Bezier interpolation: (1-t)^2 * W_start + 2t(1-t) * W_bend + t^2 * W_end
        """
        interpolated_state_dict = {}
        for (name, p_start), (_, p_end) in zip(self.model_start.named_parameters(), self.model_end.named_parameters()):
            if p_start.requires_grad:
                p_bend = self.bend_params[name.replace('.', '_')]
                
                # Bezier formula
                weight_t = ((1 - t)**2 * p_start.data) + \
                           (2 * t * (1 - t) * p_bend) + \
                           (t**2 * p_end.data)
                interpolated_state_dict[name] = weight_t
            else:
                interpolated_state_dict[name] = p_start.data
        return interpolated_state_dict

    def forward(self, t, inputs):
        # 1. Generate the weights for position 't' on the curve
        temp_state_dict = self.get_interpolated_weights(t)
        
        # 2. Temporarily load these weights into a copy of the model
        temp_model = copy.deepcopy(self.model_start)
        temp_model.load_state_dict(temp_state_dict, strict=False)
        
        # 3. Compute loss
        outputs = temp_model(**inputs)
        return outputs.loss

def train_bezier_curve(model_start, model_end, dataloader, epochs=5, device="cuda"):
    curve_model = BezierCurveModel(model_start, model_end).to(device)
    optimizer = torch.optim.Adam(curve_model.parameters(), lr=1e-3)
    
    # Train the "bend" parameters to find the lowest possible loss along the curve
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # Sample a random point t on the curve between 0.0 and 1.0
            t = torch.rand(1).item() 
            
            optimizer.zero_grad()
            loss = curve_model(t, inputs)
            loss.backward()
            optimizer.step()
            
    return curve_model