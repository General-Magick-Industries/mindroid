use async_trait::async_trait;

use crate::auth::TokenPlacement;
use crate::{Auth, Result};

pub struct StaticAuth {
    token: String,
    placement: TokenPlacement,
}

impl StaticAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            placement: TokenPlacement::Token,
        }
    }

    /// A static credential whose token is delivered in the Centrifugo connect
    /// frame's `data` field (routing to the connect proxy) rather than the
    /// JWKS-gated top-level `token`. Used for bifrost end-user tokens.
    pub fn new_end_user(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            placement: TokenPlacement::Data,
        }
    }
}

#[async_trait]
impl Auth for StaticAuth {
    async fn get_token(&self) -> Result<String> {
        Ok(self.token.clone())
    }

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {}", self.token),
        )])
    }

    fn is_authenticated(&self) -> bool {
        true
    }

    async fn refresh(&self) -> Result<()> {
        Ok(())
    }

    fn connect_token_placement(&self) -> TokenPlacement {
        self.placement
    }
}
