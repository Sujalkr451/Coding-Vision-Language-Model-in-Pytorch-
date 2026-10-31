from clip_loss import clip_contrastive_loss

image_embeddings = image_encoder(images)
text_embeddings = text_encoder(token_ids)

loss = clip_contrastive_loss(image_embeddings, text_embeddings)

loss.backward()