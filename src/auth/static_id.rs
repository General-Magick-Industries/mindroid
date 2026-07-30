use async_trait::async_trait;

use crate::{Auth, Result};

pub struct StaticAuth {
    token: String,
}

impl StaticAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
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
}
